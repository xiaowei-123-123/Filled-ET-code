import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# Define function to calculate Delta value
def calculate_delta(t):
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return 4098 * es / (t + 237.3) ** 2

# Define function to calculate Gamma value
def calculate_gamma(p):
    return Cp * p / (ε * λ)

# Define function to calculate ETgap value
def calculate_et_gap(row, obs_row):
    et_gap_candidate = (
        obs_row['ET_obs'] *
        ((row['NETRAD_gap'] - row['G_gap']) * calculate_delta(row['TA_gap'])) /
        ((obs_row['NETRAD_obs'] - obs_row['G_obs']) * calculate_delta(obs_row['TA_obs'])) *
        ((calculate_delta(obs_row['TA_obs']) + calculate_gamma(obs_row['PA_obs'])) / 
         (calculate_delta(row['TA_gap']) + calculate_gamma(row['PA_gap']))) *
        (obs_row['Beta_star_obs'] / row['Beta_star_gap'])
    )
    return et_gap_candidate

# Dynamic window function
def find_valid_windows(idx, data, window_options):
    """
    Find valid window ranges that contain at least 5 observed values based on the window combination.
    Return all valid windows that meet the criteria.
    """
    valid_windows = []
    for window in window_options:
        # Dynamically set the window range
        front_days, back_days = window
        start_date = data.loc[idx, 'DATE'] - pd.Timedelta(days=front_days)
        end_date = data.loc[idx, 'DATE'] + pd.Timedelta(days=back_days)
        
        # Filter data within the window
        valid_le_rows = data[
            (data['DATE'] >= start_date) &
            (data['DATE'] <= end_date) &
            (data['ET_obs'].notna()) &  # Keep only non-empty LEgap values
            (data['YEAR'] == data.loc[idx, 'YEAR'])  # Restrict to the same year
        ]
        
        # If there are >= 5 non-empty values in the window, record it
        if len(valid_le_rows) >= 5:
            valid_windows.append((start_date, end_date, valid_le_rows))
    
    return valid_windows

# Process each CSV file function
def process_file(file_path, output_folder_path, window_options, max_iterations=1000):
    # Read data
    data = pd.read_csv(file_path)
    data['DATE'] = pd.to_datetime(data['TIMESTAMP']).dt.date
    data['YEAR'] = pd.to_datetime(data['TIMESTAMP']).dt.year  # Extract the year column

    # Data preprocessing, add required columns
    data['z0m'] = data['Z0M']  
    data['z0h'] = data['Z0M'] * 0.1 
    data['d'] = 6.67 * data['Z0M']
    data['d'] = data['d']
    data['u*'] = data['WS']
    data['ra'] = (np.log((Z - data['d']) / data['z0m']) * np.log((Z - data['d']) / data['z0h'])) / (k**2 * data['u*'])
    data['TA_K'] = data['TA'] + 273.110
    data['Delta'] = calculate_delta(data['TA'])
    data['Gamma'] = calculate_gamma(data['PA'])
    data['p'] = data['PA']/(0.287 * data['TA_K'])
    data['Beta_star'] = 1 / (1 + (data['p'] * Cp * data['VPD']) / (data['NETRAD'] - data['G']) / data['ra'])

    # Initialize columns
    data['fill_logo'] = False
    data['ET_filled'] = data['ET_obs']  # Store filled values here
    data['iteration_count'] = 0  # Record the number of attempts to fill each cell

    # Iterative filling logic
    for iteration in range(max_iterations):
        print(f"Processing file: {os.path.basename(file_path)}, iteration {iteration+1}")
        any_changes = False  # Flag to check if any filling operation is done

        for idx in tqdm(data[data['ET_obs'].isna()].index, desc=f"Iteration {iteration+1}"):
            # Dynamically find valid windows
            valid_windows = find_valid_windows(idx, data, window_options)
            
            if valid_windows:
                # Calculate ETgap value for each valid window
                et_gap_candidates = []
                for start_date, end_date, valid_le_rows in valid_windows:
                    for valid_idx in valid_le_rows.index:
                        et_gap_candidate = calculate_et_gap(data.iloc[idx], valid_le_rows.loc[valid_idx])
                        if -50 <= et_gap_candidate <= 300:  # Limit of candidate value range
                            et_gap_candidates.append(et_gap_candidate)
                
                # Imputation logic: Use the mean of all ETgap candidates from valid windows
                if et_gap_candidates:
                    mean_et_gap = np.mean(et_gap_candidates)
                    data.at[idx, 'ET_obs'] = mean_et_gap
                    data.at[idx, 'ET_filled'] = mean_et_gap
                    data.at[idx, 'fill_logo'] = True
                    any_changes = True
                else:
                    data.at[idx, 'iteration_count'] += 1
            else:
                data.at[idx, 'iteration_count'] += 1

        # If no changes were made during this iteration, stop the loop
        if not any_changes:
            break

    # Set cells that couldn't be filled within max_iterations to NaN
    data.loc[data['iteration_count'] >= max_iterations, 'ET_obs'] = np.nan
    data.loc[data['iteration_count'] >= max_iterations, 'fill_logo'] = False

    # Save processed data
    output_file_path = os.path.join(output_folder_path, os.path.basename(file_path))
    data.to_csv(output_file_path, index=False)
    print(f"File {os.path.basename(file_path)} processing completed.")

# Constants
Cp = 1.004  # kJ/kg·K
λ = 2.45  # kJ/kg
ε = 0.622
k = 0.41
Z = 10

# Input and output folder paths
input_folder_path = r''
output_folder_path = r''

# Create output folder if not exists
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

window_options = [
    (0, 6),  # 0 days before and 6 days after
    (1, 5),  # 1 day before and 5 days after
    (2, 4),  # 2 days before and 4 days after
    (3, 3),  # 3 days before and 3 days after
    (4, 2),  # 4 days before and 2 days after
    (5, 1),  # 5 days before and 1 day after
    (6, 0),  # 6 days before and current day
]

# Iterate over each CSV file
csv_files = glob.glob(os.path.join(input_folder_path, '*.csv'))  # Get all .csv files in the folder
if not csv_files:
    print("No CSV files found, please check the input folder path.")
else:
    for file_path in csv_files:
        process_file(file_path, output_folder_path, window_options)

print("All files have been processed.")

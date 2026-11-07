import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 320.0 # 300s limit, keep 20s margin

def heuristics(demands, capacity):
    # Check time
    if time.time() - START_TIME > TIME_LIMIT:
        return np.ones((len(demands), len(demands))) * 1e-9

    n = len(demands)
    # Convert to float array to be safe
    d = np.array(demands, dtype=np.float64)
    
    # Calculate sums
    sums = d[:, None] + d[None, :]
    
    # Create mask for valid pairs (fit in capacity)
    valid = sums <= capacity
    
    # Initialize eta with small positive values for invalid entries
    eta = np.full((n, n), 1e-9, dtype=np.float64)
    
    if np.any(valid):
        # Calculate ratio for valid pairs
        # Using a power less than 1 (e.g., 0.5) to flatten the distribution
        # This is based on the observation that the linear heuristic (power 1)
        # outperformed the cubic (power 3) and inverse-gap heuristics.
        # Flattening further may allow pheromone to guide the search more effectively
        # by not over-committing to tight fits.
        ratio = np.zeros((n, n), dtype=np.float64)
        ratio[valid] = sums[valid] / capacity
        
        # Apply power 0.5
        eta[valid] = ratio ** 0.5
        
        # Safety check for non-finite values
        eta[~np.isfinite(eta)] = 1e-9
        
    return eta

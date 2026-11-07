import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 320.0  # 300s limit, keep 20s margin

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
        # Calculate gap for valid pairs
        gaps = capacity - sums[valid]
        
        # Primary signal: inverse gap (sharper best-fit preference)
        # Using a small epsilon to avoid division by zero and create a strong peak for exact fits
        inv_gap = 1.0 / (gaps + 1e-6)
        
        # Secondary signal: fill ratio power (smoother, adds secondary info)
        ratio = sums[valid] / capacity
        fill_score = ratio ** 1.5
        
        # Calculate degree of each item j (number of valid partners)
        # valid is (n, n), sum over axis 0 gives count for each j
        deg = np.sum(valid, axis=0)
        
        # Scarcity weight: items with fewer partners (larger items) get higher weight
        # This encourages placing "hard" items
        scarcity_weight = 1.0 / (deg + 1.0)
        
        # Apply scarcity weight to the heuristic matrix
        # We need to multiply column j by scarcity_weight[j]
        # Since we are filling eta[valid], we can't easily do this in-place with boolean indexing
        # unless we construct the full matrix first.
        
        # Let's construct the base heuristic for valid entries
        base_heur = np.zeros((n, n), dtype=np.float64)
        base_heur[valid] = inv_gap * fill_score
        
        # Apply scarcity weight per column
        eta = base_heur * scarcity_weight[None, :]
        
        # Ensure invalid entries remain 1e-9 (they are 0 in base_heur, so they become 0, we need to reset)
        eta[~valid] = 1e-9
        
        # Safety check for non-finite values
        eta[~np.isfinite(eta)] = 1e-9
        
        # Ensure all values are at least 1e-9
        eta = np.maximum(eta, 1e-9)
        
    return eta

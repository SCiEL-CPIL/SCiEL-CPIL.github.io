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
    
    # Calculate sums for all pairs
    sums = d[:, None] + d[None, :]
    
    # Create mask for valid pairs (fit in capacity)
    valid = sums <= capacity
    
    # Initialize eta with small positive values for invalid entries
    eta = np.full((n, n), 1e-9, dtype=np.float64)
    
    if np.any(valid):
        # Calculate gap for valid pairs
        gaps = capacity - sums
        
        # Determine the minimum item size to identify "bin closing" moves
        d_min = np.min(d)
        
        # A move closes a bin if the remaining gap is less than the smallest item
        # i.e., no other item can fit into the remaining space
        is_closed = gaps < d_min
        
        # Base heuristic: Inverse gap
        # We add a small epsilon to avoid division by zero and to smooth the function
        # For closed bins, the gap is small, so 1/gap is large.
        # For open bins, the gap is larger, so 1/gap is smaller.
        # We want to strongly prefer closed bins.
        
        # Let's use a multiplicative factor to boost closed bins
        # If gap < d_min, we are in the "closed" category.
        # We want the heuristic to be high.
        
        # Strategy:
        # 1. Calculate a base score based on fill ratio or inverse gap.
        # 2. Apply a large multiplier if the bin is closed.
        
        # Using inverse gap (sharper preference for tight fits)
        # To prevent overflow or extreme values, we can clip or normalize.
        # However, ACO usually handles varying scales if pheromone is also scaled.
        # Here pheromone starts at 1.
        
        # Let's try:
        # eta[i, j] = 1 / (gap + 1) * (100 if closed else 1)
        
        # This ensures that any closed bin move is 100x more attractive than an open bin move
        # with the same gap. Since closed bins have small gaps, their 1/gap is already large.
        # This amplifies the signal significantly.
        
        base_score = 1.0 / (gaps + 1.0)
        
        # Apply the bin-closing boost
        # We only apply this to valid entries
        # is_closed is True for valid entries where gap < d_min
        # For invalid entries, gaps > capacity, so is_closed is False (since d_min > 0)
        # But we only update eta where valid is True.
        
        boost = np.where(is_closed, 100.0, 1.0)
        
        eta_vals = base_score * boost
        
        # Assign to eta where valid
        eta[valid] = eta_vals[valid]
        
        # Safety check for non-finite values
        eta[~np.isfinite(eta)] = 1e-9
        
        # Ensure all values are at least 1e-9
        eta = np.maximum(eta, 1e-9)
        
    return eta

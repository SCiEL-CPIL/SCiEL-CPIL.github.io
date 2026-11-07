import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Compute sum of all pairs
    pair_sum = d[:, None] + d[None, :]
    
    # Calculate waste (remaining space in bin if only i and j are in it)
    waste = capacity - pair_sum
    
    # Determine valid pairs (those that fit)
    valid = pair_sum <= capacity + 1e-9
    
    # Initialize heuristic matrix
    h = np.zeros((n, n), dtype=float)
    
    # For valid pairs, compute attractiveness using squared ratio
    # (pair_sum / capacity)^2 strongly rewards tight packing
    # Combined with inverse waste for additional discrimination
    waste_clipped = np.maximum(waste, 0.0)
    
    # Ratio term: (pair_sum / capacity)^2 - rewards pairs that fill the bin
    ratio = pair_sum / capacity
    ratio_term = ratio ** 2
    
    # Inverse waste term: 1 / (waste + 1) - additional penalty for loose packing
    inv_waste_term = 1.0 / (waste_clipped + 1.0)
    
    # Combine both terms for the heuristic
    # The squared ratio provides strong signal for tight packing
    h[valid] = ratio_term[valid] + 0.5 * inv_waste_term[valid]
    
    # Check deadline - if exceeded, return what we have (which should be complete)
    if time.time() > deadline:
        # Ensure non-positive values become 1e-9 as required
        h[~np.isfinite(h) | (h <= 0)] = 1e-9
        return h
    
    # The prompt states: "Heuristic entries that are not positive become 1e-9"
    # Ensure all non-positive or non-finite values are set to 1e-9
    h[~np.isfinite(h) | (h <= 0)] = 1e-9
    
    return h

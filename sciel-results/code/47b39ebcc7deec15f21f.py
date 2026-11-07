import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = 330.0  # Safety margin from 300s limit, though this is fast
    
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Calculate pair sums
    sums = demands[:, None] + demands[None, :]
    
    # Identify valid pairs (sum <= capacity)
    valid = sums <= capacity
    
    # Calculate gap for valid pairs
    # gap = capacity - sum
    gaps = capacity - sums
    
    # Initialize heuristic with a very small value
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Extract indices for valid pairs
    i_idx, j_idx = np.where(valid)
    
    if len(i_idx) == 0:
        # No valid pairs, return base matrix
        return eta
        
    d_j = demands[j_idx]
    waste = gaps[valid]
    
    # Refinement from Inspiration 2: Scale the waste by the minimum demand.
    # This normalizes the penalty so that "small" waste relative to item sizes
    # is not over-penalized, allowing the heuristic to distinguish between
    # slightly loose and very loose packs more effectively.
    d_min = np.min(demands)
    
    # Avoid division by zero if d_min is 0 (though demands are 20-99, so safe)
    if d_min <= 0:
        d_min = 1.0
        
    scaled_waste = waste / d_min
    
    # Heuristic formula:
    # Numerator d_j rewards placing larger items (which are harder to pack).
    # Denominator penalizes waste. Squared penalty makes the gradient steeper
    # for larger wastes, but scaling prevents saturation for small wastes.
    scores = d_j / (1.0 + scaled_waste * scaled_waste)
    
    # Assign to matrix
    eta[valid] = scores
    
    # Diagonal: set to 1e-9 (cannot place item after itself)
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure no non-finite values
    eta[~np.isfinite(eta)] = 1e-9
    
    # Deadline check (unlikely to be hit for n=120)
    if time.time() > deadline:
        # Return best result found (which is the current eta)
        pass
        
    return eta

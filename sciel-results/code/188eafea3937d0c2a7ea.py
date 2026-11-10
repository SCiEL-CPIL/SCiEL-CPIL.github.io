import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands)
    
    # Calculate pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Calculate gaps (remaining space)
    gaps = capacity - pair_sums
    
    # Heuristic: Inverse of gap
    # If pair fits (gap >= 0), eta = 1 / (gap + 1)
    # If pair doesn't fit, eta = 1e-9
    # Using 1.0 as offset to avoid division by zero and to scale values reasonably
    # The +1 ensures that even a perfect fit (gap=0) has a finite value, 
    # and it prevents the heuristic from dominating too aggressively if gaps are very small.
    
    valid = pair_sums <= capacity
    eta = np.zeros((n, n), dtype=float)
    
    # Compute for valid pairs
    # We use np.where to avoid division by zero warnings or issues, 
    # but we only assign to valid indices.
    # To be safe and efficient:
    eta[valid] = 1.0 / (gaps[valid] + 1.0)
    
    # Set invalid pairs to 1e-9
    eta[~valid] = 1e-9
    
    return eta

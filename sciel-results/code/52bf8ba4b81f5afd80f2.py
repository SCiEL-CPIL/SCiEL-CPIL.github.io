import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # Create pair sum matrix
    pair_sum = demands[:, None] + demands[None, :]
    
    # Calculate remaining space (gap)
    gap = capacity - pair_sum
    
    # Initialize heuristic matrix with a very small value
    H = np.full((n, n), 1e-9)
    
    # Valid pairs are those where gap >= 0
    valid = gap >= 0
    
    if not np.any(valid):
        return H

    # New Heuristic: Exponential Tightness
    # H_ij = exp(-alpha * gap_ij)
    # This strongly favors pairs that fill the bin completely (gap=0)
    # and penalizes leftover space exponentially.
    alpha = 0.1
    
    # Calculate raw values for valid entries
    raw_values = np.exp(-alpha * gap)
    
    # Apply only to valid entries
    H[valid] = raw_values[valid]
    
    return H

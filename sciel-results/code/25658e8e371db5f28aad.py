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
    
    # For valid pairs, use inverse gap heuristic
    # Add 1 to avoid division by zero and to scale
    # H = 1 / (gap + 1)
    H[valid] = 1.0 / (gap[valid] + 1.0)
    
    return H

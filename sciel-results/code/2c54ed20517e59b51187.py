import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # demands is a float array
    # capacity is float
    pair = demands[:, None] + demands[None, :]
    gap = capacity - pair
    
    # Initialize with 1e-9
    H = np.full((n, n), 1e-9)
    
    # Mask for valid pairs
    mask = pair <= capacity
    
    # New Heuristic: Size-Weighted Complementarity
    # H = (d_i * d_j) / (gap + 1)
    # This approach favors pairs of larger items (which are harder to place)
    # that also have a small gap (fit well).
    d_prod = demands[:, None] * demands[None, :]
    
    H[mask] = d_prod[mask] / (gap[mask] + 1.0)
    
    # Normalize to [0, 1] for numerical stability and standardization
    # This does not change relative probabilities in ACO
    max_val = H.max()
    if max_val > 0:
        H = H / max_val
        
    return H

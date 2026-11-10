import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # demands is a float array
    # capacity is float
    pair = demands[:, None] + demands[None, :]
    gap = capacity - pair
    
    # Heuristic is high if gap is small (pair fills bin well)
    # Only valid if pair <= capacity
    # If pair > capacity, gap < 0, we want 1e-9
    # If pair <= capacity, gap >= 0
    
    # Use 1 / (gap + 1) to avoid division by zero and keep values bounded
    # This function is monotonically decreasing with gap
    # So it favors smaller gaps (larger sums)
    
    # Initialize with 1e-9
    H = np.full((n, n), 1e-9)
    
    # Mask for valid pairs
    mask = pair <= capacity
    
    # Calculate heuristic for valid pairs
    # gap is non-negative where mask is True
    # H = 1 / (gap + 1)
    H[mask] = 1.0 / (gap[mask] + 1.0)
    
    return H

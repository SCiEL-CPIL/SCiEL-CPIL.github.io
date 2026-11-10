import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing task.
    
    Args:
        demands: list or array of item sizes (floats).
        capacity: bin capacity (float).
        
    Returns:
        numpy array of shape (n, n) with heuristic values.
    """
    n = len(demands)
    d = np.array(demands, dtype=np.float64)
    
    # Calculate pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Calculate waste
    waste = capacity - pair_sums
    
    # Initialize heuristic matrix
    # We use 0 for non-fitting items, which will be clamped to 1e-9 by the evaluator
    H = np.zeros((n, n), dtype=np.float64)
    
    # Mask for items that fit together
    fits = pair_sums <= capacity
    
    if np.any(fits):
        # Add a small epsilon to avoid division by zero for perfect fits
        eps = 1e-6
        valid_waste = waste[fits] + eps
        
        # Squared inverse waste heuristic
        # This strongly rewards pairs that leave very little space in the bin
        H[fits] = 1.0 / (valid_waste ** 2)
        
        # Ensure all values are finite and positive
        # 1/w^2 is always positive and finite for w >= eps
        # But just in case of any numerical weirdness
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9

    return H

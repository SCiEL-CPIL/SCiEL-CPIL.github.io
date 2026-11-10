import numpy as np
import time

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
    
    # Calculate waste (remaining space in bin after placing i and j)
    waste = capacity - pair_sums
    
    # Initialize heuristic matrix
    # We use 0 for non-fitting items, which will be clamped to 1e-9 by the evaluator
    H = np.zeros((n, n), dtype=np.float64)
    
    # Mask for items that fit together
    fits = pair_sums <= capacity
    
    if np.any(fits):
        # Use a composite heuristic:
        # 1. Reward tight fits (small waste) with 1/waste
        # 2. Penalize very large items that leave little room for future items
        #    but reward medium items that fit well with others.
        #    Actually, the key insight for bin packing is that we want to fill bins as full as possible.
        #    A pair (i, j) is good if d[i] + d[j] is close to capacity.
        #    So 1/waste is a good measure.
        
        # However, we should also consider that placing two very small items together
        # leaves a lot of waste, which is bad.
        # But 1/waste already handles this: if waste is large, 1/waste is small.
        
        # Let's try a power law to amplify the difference between good and bad fits.
        # 1/waste^p with p > 1 makes tight fits much more attractive.
        # The current score is 1.046 with p=1.
        # Let's try p=2 or p=3.
        
        # Also, we can add a term that favors pairs where the sum is closer to capacity.
        # waste = capacity - (d[i] + d[j])
        # So minimizing waste is equivalent to maximizing d[i] + d[j].
        
        eps = 1e-6
        valid_waste = waste[fits] + eps
        
        # Try p=2 to strongly prefer tight fits
        H[fits] = 1.0 / (valid_waste ** 2)
        
        # Ensure all values are finite and positive
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9

    return H

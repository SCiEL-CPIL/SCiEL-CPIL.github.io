import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the Ant Colony Optimization.
    
    Args:
        demands: A list or array of item weights (floats).
        capacity: The capacity of the bins.
        
    Returns:
        A 2D numpy array of shape (n, n) where n is the number of items.
        Entry (i, j) is the attractiveness of placing item j after item i.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Pairwise sums of weights
    sums = demands[:, None] + demands[None, :]
    
    # Initialize with a very small positive value for invalid pairs
    H = np.full((n, n), 1e-9, dtype=float)
    
    # Identify valid pairs that fit in a bin
    valid_mask = sums <= capacity
    
    if not np.any(valid_mask):
        return H
        
    # Calculate waste for valid pairs
    # waste = capacity - sums
    # For valid pairs, waste >= 0
    
    # Fill ratio: how much of the capacity is used
    # We use a high power to strongly prefer pairs that fill the bin
    fill_ratio = sums / capacity
    base_score = np.power(fill_ratio, 15)
    
    # Bonus for tight fits (small waste)
    # This encourages perfect or near-perfect complements
    waste = capacity - sums
    bonus_score = 2.0 / (waste + 1.0)
    
    # Combine scores for valid pairs
    H[valid_mask] = base_score[valid_mask] + bonus_score[valid_mask]
    
    # Ensure all values are positive and finite
    # np.power and division should be safe here given the constraints
    # but we enforce positivity as per task description
    H = np.maximum(H, 1e-9)
    
    return H

import numpy as np

def heuristics(prizes, weights):
    """
    Calculates attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Args:
        prizes: numpy array of shape (5, 100), uniform in [0, 1].
        weights: numpy array of shape (5, 100), uniform in [0, 1].
        
    Returns:
        numpy array of shape (100,), where each entry is the heuristic score for an item.
    """
    # 1. Calculate the total prize for each item (sum across the 5 constraints)
    # Shape: (100,)
    total_prizes = np.sum(prizes, axis=0)
    
    # 2. Calculate the 'bottleneck' cost for each item.
    # Using the L2 norm (Euclidean distance) of the weight vector.
    # This acts as a 'soft' maximum, smoothing out the penalty of the single heaviest constraint
    # while still prioritizing items that are light overall.
    # Shape: (100,)
    weight_norms = np.linalg.norm(weights, axis=0)
    
    # 3. Compute the heuristic: Prize / Cost
    # Add a small epsilon to prevent division by zero
    eps = 1e-9
    heuristic = total_prizes / (weight_norms + eps)
    
    # 4. Ensure validity:
    # - Replace non-finite values (inf, nan) with a small positive number.
    # - Ensure all values are positive (as required by ACO probability calculation).
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic

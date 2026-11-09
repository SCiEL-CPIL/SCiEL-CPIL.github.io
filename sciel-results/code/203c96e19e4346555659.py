import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # bins is an array of remaining capacities
    # Calculate new remaining capacities if item is placed in each bin
    new_remains = bins - item
    
    # Only consider bins where new_remain >= 0
    valid_mask = new_remains >= 0
    
    # If no valid bins, return -inf for all (or handle error, but contract says "enough remaining capacity")
    # The contract says "scores each bin with enough remaining capacity".
    # If a bin doesn't have enough, it shouldn't be chosen.
    # We should return -inf for invalid bins so they are never chosen.
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores
        
    # Calculate the mean of the *valid* new remainings? 
    # Or the mean of the *current* bins?
    # Let's use the mean of the current valid bins' remaining capacities as a target.
    # Actually, using the mean of the *resulting* remainings is circular.
    # Let's use the mean of the *current* remaining capacities of valid bins.
    
    current_mean = np.mean(bins[valid_mask])
    
    # Score is negative absolute difference between new remaining and current mean
    # We want to minimize the difference, so we maximize -difference
    diffs = np.abs(new_remains[valid_mask] - current_mean)
    scores[valid_mask] = -diffs
    
    return scores

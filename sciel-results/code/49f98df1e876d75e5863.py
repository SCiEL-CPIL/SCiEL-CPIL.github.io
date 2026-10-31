import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    C = 100.0
    remaining = bins - item
    
    # Only consider bins with enough capacity
    valid_mask = remaining >= 0
    
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores

    r_valid = remaining[valid_mask]
    
    # Polarization score: distance from half-capacity
    # We want to maximize distance from 50
    base_score = np.abs(r_valid - C / 2.0)
    
    # Tie-breaking logic
    # If item is small, we prefer to fill up bins (smaller remaining) to consolidate.
    # If item is large, we prefer to keep bins empty (larger remaining) to avoid fragmentation.
    # We can achieve this by adding a small epsilon term.
    
    if item < C / 2.0:
        # Prefer smaller remaining (Best Fit)
        # Subtract a small fraction of remaining capacity
        scores_valid = base_score - 1e-9 * r_valid
    else:
        # Prefer larger remaining (Worst Fit)
        # Add a small fraction of remaining capacity
        scores_valid = base_score + 1e-9 * r_valid
        
    scores[valid_mask] = scores_valid
    return scores

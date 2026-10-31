import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Worst-Fit.
    We prioritize bins with the LARGEST remaining capacity after placing the item.
    This is structurally different from Best-Fit (which minimizes remaining capacity).
    Worst-Fit tends to keep bins with large gaps, which might be beneficial for
    packing future smaller items in distributions with a heavy tail of small items
    (like Weibull with k < 1), preventing fragmentation into many small unusable gaps.
    """
    # Compute remaining capacity after placing the item
    remaining_after = bins - item
    
    # Only consider bins with enough capacity
    valid_mask = remaining_after >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    # For valid bins, use Worst-Fit: prefer largest remaining_after
    # Score is the remaining capacity itself.
    scores[valid_mask] = remaining_after[valid_mask]
    
    return scores

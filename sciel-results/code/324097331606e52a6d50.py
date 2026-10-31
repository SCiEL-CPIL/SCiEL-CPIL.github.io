import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin for the given item.
    Higher score is better.
    
    Uses a hybrid strategy based on item size:
    - For large items (> 40): Worst Fit - maximize remaining capacity to keep bins open
    - For small items (<= 40): Best Fit - minimize remaining capacity to consolidate
    """
    n_bins = len(bins)
    scores = np.full(n_bins, -1e9, dtype=np.float64)
    
    # Find bins that can accommodate the item
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
        
    valid_remaining = bins[valid_mask]
    
    # For large items, use Worst Fit (maximize remaining space)
    # This keeps bins with more free space available for other large items
    # For small items, use Best Fit (minimize remaining space)
    # This consolidates small items to reduce fragmentation
    
    item_ratio = item / 100.0
    
    if item_ratio > 0.4:
        # Large item: Worst Fit - score is the remaining capacity after placement
        # Higher remaining is better (we want to preserve space for future large items)
        remaining_after = valid_remaining - item
        scores[valid_mask] = remaining_after
    else:
        # Small item: Best Fit - score is negative remaining capacity after placement
        # Lower remaining is better (we want to fill bins tightly)
        remaining_after = valid_remaining - item
        scores[valid_mask] = -remaining_after
    
    return scores

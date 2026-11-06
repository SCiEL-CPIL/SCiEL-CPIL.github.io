import numpy as np

# State for the running mean
_item_count = 0
_item_sum = 0.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _item_count, _item_sum
    
    # Update statistics
    _item_count += 1
    _item_sum += item
    mean_item = _item_sum / _item_count
    
    # Identify valid bins (enough capacity)
    remaining = bins - item
    valid_mask = remaining >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores

    # For valid bins, calculate the remaining space after placing the item
    next_remaining = remaining[valid_mask]
    
    # The heuristic: minimize the absolute difference between the new remaining space
    # and the expected size of the next item (approximated by the running mean).
    # This tries to "fit" the next item into the gap left by the current one.
    target_gap = mean_item
    distances = np.abs(next_remaining - target_gap)
    
    # We want to minimize distance, so score is negative distance
    scores[valid_mask] = -distances
    
    return scores

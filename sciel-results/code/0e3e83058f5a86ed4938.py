import numpy as np

# We assume capacity is 100 as per task description
CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining space
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Find valid indices (bins with enough remaining capacity)
    valid_mask = remaining >= 0
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) == 0:
        return scores

    # Next-Item Awareness Heuristic
    # We estimate the expected size of the next item to be similar to the current item (stationarity).
    # We want to leave a remaining space in the bin that can accommodate this next item.
    # Let R_b be the remaining space before placing the item.
    # After placing the item, the new remaining space is R'_b = R_b - item.
    # We want R'_b to be close to the expected next item size (item).
    # So, R_b - item ~= item  =>  R_b ~= 2 * item.
    
    # Score is the negative absolute difference between the current remaining space
    # and the target remaining space (2 * item).
    # Higher scores (closer to 0) are preferred.
    
    target_remaining = 2.0 * item
    scores[valid_idx] = -np.abs(remaining[valid_idx] - target_remaining)
    
    return scores

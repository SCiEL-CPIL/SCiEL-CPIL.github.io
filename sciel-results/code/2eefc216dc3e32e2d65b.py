import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    scores = np.full_like(bins, -np.inf)
    
    # Best Fit strategy:
    # We want to select the bin with the smallest remaining capacity
    # that can accommodate the item.
    # Score = -remaining capacity.
    # Higher score means smaller remaining capacity (tighter fit).
    scores[valid_mask] = -remaining[valid_mask]
    
    return scores

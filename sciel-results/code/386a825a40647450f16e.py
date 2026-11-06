import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Valid bins
    valid = bins >= item
    
    if not np.any(valid):
        return scores
    
    # Worst-Fit Strategy:
    # Prefer the bin with the largest remaining capacity that still fits the item.
    # This balances the load across bins and preserves large contiguous spaces
    # for potentially large future items.
    # To achieve this with a "highest score wins" mechanism:
    # Assign a score directly proportional to the remaining capacity.
    # score = remaining_capacity.
    # The bin with the largest remaining capacity will have the highest score.
    
    valid_indices = np.where(valid)[0]
    scores[valid_indices] = bins[valid_indices]
    
    return scores

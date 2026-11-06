import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Valid bins
    valid = bins >= item
    
    if not np.any(valid):
        return scores
    
    # Best-Fit Strategy:
    # Prefer the bin with the smallest remaining capacity that still fits the item.
    # This minimizes wasted space in the chosen bin.
    # To achieve this with a "highest score wins" mechanism:
    # Assign a score inversely proportional to the remaining capacity.
    # score = -remaining_capacity.
    # The bin with the smallest remaining capacity will have the largest (least negative) score.
    
    valid_indices = np.where(valid)[0]
    scores[valid_indices] = -bins[valid_indices]
    
    return scores

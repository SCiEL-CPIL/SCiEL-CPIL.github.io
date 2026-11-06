import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    valid = bins >= item
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
    
    # Adaptive: best-fit for large items, worst-fit for small items
    if item > 50.0:
        # Best-fit: minimize remaining space after placement
        scores = np.where(valid, -(bins - item), -np.inf)
    else:
        # Worst-fit: maximize remaining space after placement
        scores = np.where(valid, bins - item, -np.inf)
    return scores

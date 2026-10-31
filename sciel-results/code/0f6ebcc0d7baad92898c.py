import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    scores = np.full_like(bins, -np.inf)
    
    # Smooth transition between Best Fit and Worst Fit
    # Map item size to a weight in [0, 1]
    # Use smoothstep: 0 for item <= low, 1 for item >= high
    low = 30.0
    high = 70.0
    if item <= low:
        w = 0.0
    elif item >= high:
        w = 1.0
    else:
        t = (item - low) / (high - low)
        w = t * t * (3.0 - 2.0 * t)  # smoothstep
    
    # Blend: score = (2w - 1) * remaining
    # w=0 -> score = -remaining (Best Fit)
    # w=1 -> score = remaining (Worst Fit)
    scores[valid_mask] = (2.0 * w - 1.0) * remaining[valid_mask]
    
    return scores

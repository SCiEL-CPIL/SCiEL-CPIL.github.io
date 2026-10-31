import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the arriving item.
    The item goes to the highest-scoring bin.
    
    Strategy:
    - Smooth blend of Best-Fit and Worst-Fit.
    - Best-Fit (minimize remaining) is favored for small items to pack tightly.
    - Worst-Fit (maximize remaining) is favored for large items to preserve space.
    - Uses a quadratic weight for Worst-Fit to bias strongly towards Best-Fit,
      as pure Best-Fit performs well on Weibull distributions.
    """
    remaining = bins - item
    
    # Valid bins must have remaining >= 0
    valid = bins >= item
    
    # Weight for Worst-Fit component
    # w=0 -> Best Fit, w=1 -> Worst Fit
    # Quadratic growth ensures w is small for most items
    w = (item / 100.0) ** 2
    
    # Score = (1-w) * BestFitScore + w * WorstFitScore
    # BestFitScore = -remaining
    # WorstFitScore = remaining
    # Score = (1-w)*(-r) + w*(r) = r * (w - (1-w)) = r * (2w - 1)
    scores = remaining * (2 * w - 1)
    
    # Mask invalid bins
    scores = np.where(valid, scores, -np.inf)
    
    return scores

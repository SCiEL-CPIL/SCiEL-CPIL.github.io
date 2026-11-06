import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Strategy: Best-Fit with Smooth Fragment Penalty.
    Base score is -remaining (Best-Fit).
    A smooth penalty is added for bins that would leave a very small remaining capacity,
    to avoid creating unusable fragments.
    """
    t0 = time.time()
    
    # Bins that can fit the item
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    if np.any(valid):
        rem = remaining[valid]
        
        # Smooth fragment penalty
        # Tuned to be monotonically decreasing (beta * gamma < 1)
        # to ensure we always prefer smaller remainders (Best Fit behavior)
        # without creating a local preference for "middle" remainders.
        beta = 1.0
        gamma = 0.5
        
        # Calculate penalty
        penalty = beta * np.exp(-gamma * rem)
        
        scores[valid] = -rem - penalty
    
    # Deadline guard
    if time.time() - t0 > 110:
        # Fallback: pure best-fit
        scores = np.where(valid, -remaining, -np.inf)
    
    return scores

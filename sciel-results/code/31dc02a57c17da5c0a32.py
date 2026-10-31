"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Best Fit with Dead Space Penalty.
    We prefer bins that minimize the remaining capacity (Best Fit),
    but we penalize bins that would leave a "dead space" gap 
    (smaller than a threshold) that is likely too small to be filled 
    by future items. This encourages leaving slightly larger, 
    more useful gaps.
    """
    # Deadline check: fall back to pure Best Fit
    if time.time() > DEADLINE - SAFETY_MARGIN:
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=np.float64)
        if np.any(feasible):
            scores[feasible] = -(bins[feasible] - item)
        return scores

    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=np.float64)

    remaining = bins[feasible] - item
    
    # Base score: Best Fit (minimize remaining)
    base_score = -remaining
    
    # Dead Space Penalty
    # If the remaining space is below a threshold, apply a penalty
    # to discourage creating tiny, unusable gaps.
    # Threshold 10.0 is a heuristic guess for Weibull items with C=100.
    # Slope 2.0 makes the penalty significant enough to change the argmax
    # from the absolute tightest fit to a slightly looser fit if the
    # tightest fit leaves a gap < 10.
    threshold = 10.0
    slope = 2.0
    
    # Penalty is 0 if remaining >= threshold
    # Penalty is (threshold - remaining) * slope if remaining < threshold
    penalty = np.maximum(0.0, threshold - remaining) * slope
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    scores[feasible] = base_score - penalty

    return scores
# EVOLVE-BLOCK-END

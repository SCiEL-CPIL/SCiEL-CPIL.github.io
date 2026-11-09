"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin for 150s limit

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Median-Fit with Best-Fit tiebreaker.
    We aim to keep the remaining capacities of all bins balanced around the median.
    This reduces fragmentation by avoiding extreme disparities in remaining space.
    
    1. Calculate remaining capacity for each bin.
    2. Filter valid bins (remaining >= item).
    3. Calculate the median of the remaining capacities of valid bins.
    4. Score each valid bin by the negative squared distance of its new remaining
       capacity from the median.
    5. Add a small penalty for larger remaining capacities (Best-Fit tiebreaker)
       to encourage consolidation in case of ties.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to simple Best Fit
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full(len(bins), -np.inf)
    
    if np.any(valid):
        valid_remaining = remaining[valid]
        
        # Use median as the target for balance
        # Median is robust to outliers (very full or very empty bins)
        target = np.median(valid_remaining)
        
        # Score: negative squared distance from target
        # Maximizing this score minimizes the deviation from the median
        deviations = valid_remaining - target
        scores[valid] = -(deviations ** 2)
        
        # Tiebreaker: if deviations are equal (or very close), prefer smaller remaining
        # This mimics Best-Fit behavior for consolidation
        scores[valid] -= 1e-6 * valid_remaining
        
    return scores


# EVOLVE-BLOCK-END

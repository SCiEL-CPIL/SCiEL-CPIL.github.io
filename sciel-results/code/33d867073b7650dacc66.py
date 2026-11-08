"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for deadline tracking
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of one score per bin (higher = better).
    
    Strategy: Best-fit with gap-awareness for Weibull-distributed items.
    - Prefer bins where item fits with minimal leftover (best-fit).
    - Penalize creating tiny gaps that won't fit future Weibull items.
    - For very small items, prefer fuller bins to reduce fragmentation.
    """
    global _start_time
    
    # Deadline guard: if we're running long, fall back to simple best-fit
    elapsed = time.time() - _start_time
    if elapsed > 145.0:
        # Simple best-fit as fallback
        remaining = bins - item
        mask = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        scores[mask] = -remaining[mask]
        return scores

    remaining = bins - item
    mask = remaining >= 0
    
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(mask):
        return scores
    
    rem = remaining[mask]
    
    # Base score: best-fit (less remaining = better)
    base_score = -rem
    
    # Gap penalty: if remaining is very small but > 0, it's wasted space
    # For Weibull items with capacity 100, typical items are small (1-10)
    # A gap of < 2 is likely unusable for future items
    # Penalize gaps that are "awkward" - not zero but too small to be useful
    gap_penalty = np.zeros_like(rem)
    # Gaps between 0 and ~3 are likely wasted (Weibull min is often > 1)
    # Use a smooth penalty: gaps in (0, 5) get penalized proportionally
    small_gap_mask = (rem > 0) & (rem < 5.0)
    gap_penalty[small_gap_mask] = rem[small_gap_mask] * 0.5  # extra penalty for small gaps
    
    # Utilization bonus: for very small items, prefer bins that are already full
    # This reduces the number of partially-filled bins
    if item < 2.0:
        # Strong preference for fuller bins
        utilization_bonus = -rem * 0.3
    elif item < 5.0:
        utilization_bonus = -rem * 0.1
    else:
        utilization_bonus = np.zeros_like(rem)
    
    # Combined score
    final_scores = base_score - gap_penalty + utilization_bonus
    
    scores[mask] = final_scores
    
    return scores
# EVOLVE-BLOCK-END

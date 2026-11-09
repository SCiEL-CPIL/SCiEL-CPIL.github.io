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
    
    # Smooth gap penalty using a soft-min surrogate.
    # The non-smooth objective "wasted gap" is approximated by:
    #   gap_penalty = -log(1 + exp(-k*(rem - t))) / k  (smooth min(rem, t) style)
    # This smoothly transitions from penalizing small gaps to accepting larger ones.
    # t = threshold where gap becomes "usable" for future Weibull items (~2.0)
    # k = sharpness of the transition
    t = 2.0
    k = 3.0
    # Smooth approximation of min(rem, t): for rem < t, penalty ~ rem; for rem > t, penalty ~ t
    # Using: penalty = t - (1/k)*log(1 + exp(-k*(rem - t)))
    # For rem << t: exp(-k*(rem-t)) = exp(k*(t-rem)) is large, log ~ k*(t-rem), penalty ~ t - (t-rem) = rem
    # For rem >> t: exp(-k*(rem-t)) ~ 0, log ~ 0, penalty ~ t
    # This gives a smooth penalty that's ~rem for small gaps and saturates at t for large gaps
    gap_penalty = t - (1.0/k) * np.log1p(np.exp(-k*(rem - t)))
    
    # Utilization bonus: for small items, prefer bins that are already full
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

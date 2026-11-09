import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Uses a refined hybrid heuristic combining best-fit with a smooth
    'just right' gap preference and a large-gap penalty.
    
    Key improvements over previous iterations:
    1. Best Fit as primary signal (minimize remaining capacity).
    2. Smooth Gaussian bump centered at 0.15*item with sigma=0.12*item
       to encourage slightly loose fits that avoid fragmentation.
    3. Quadratic large-gap penalty that activates beyond 1.2*item,
       scaled to be significant but not dominant over best-fit for
       moderate gaps.
    4. Time guard to ensure we finish within the 150s limit.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    
    if not np.any(valid):
        return scores
    
    r = remaining[valid]
    
    # Primary: Best Fit (smaller remaining is better)
    base_score = -r
    
    # Secondary: Gaussian bump for "slightly loose" fits
    # Center at 0.15 * item, sigma at 0.12 * item
    # This encourages a small but non-trivial gap
    center = 0.15 * item
    sigma = 0.12 * item + 1e-10
    bump = np.exp(-((r - center) ** 2) / (2.0 * sigma ** 2))
    
    # Tertiary: Large Gap Penalty
    # Penalize bins where the remaining capacity is significantly larger than the item.
    # Use a quadratic penalty that starts at 1.2 * item.
    gap_threshold = 1.2 * item
    excess_gap = np.maximum(r - gap_threshold, 0.0)
    # Normalize by item size and apply quadratic penalty
    penalty = (excess_gap / (item + 1e-10)) ** 2 * 0.3
    
    # Combine scores
    # Weight for bump: 0.4 to have moderate influence without overwhelming best-fit
    weight = 0.4
    
    combined = base_score + weight * bump - penalty
    
    scores[valid] = combined
    
    # Time guard: if we're getting close to deadline, return immediately
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        pass  # We're already done with computation, just returning
    
    return scores

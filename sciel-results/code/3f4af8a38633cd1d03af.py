"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Strategy: Best-fit with harmonic-mean fragmentation penalty.
    We model bin selection as a resource fragmentation problem.
    The score penalizes both tiny leftover gaps (waste) and excessively large
    remaining spaces (risk of future fragmentation).
    
    We use a smooth surrogate based on the harmonic mean of remaining capacity
    and item size:
        H(rem, item) = 2 * rem * item / (rem + item)
    
    The harmonic mean is small when either rem or item is small, and large when
    both are large. For best-fit, we want to prefer bins where rem is small
    (so the item fits tightly). However, if rem is too small relative to item,
    it might not fit. Since we only consider valid bins (rem >= 0), and item > 0,
    we want to maximize the "fit quality".
    
    Actually, let's think differently. The directive says: "scoring bins with the
    negative of the harmonic mean of their remaining capacity and the incoming item size".
    
    So score = -H(rem, item) = -2 * rem * item / (rem + item)
    
    Let's analyze this:
    - If rem is very small (e.g., rem ≈ 0), H ≈ 0, so score ≈ 0.
    - If rem is very large (e.g., rem >> item), H ≈ 2*item, so score ≈ -2*item.
    - If rem ≈ item, H = 2*item*item/(2*item) = item, so score = -item.
    
    Wait, this means:
    - rem = 0: score = 0
    - rem = item: score = -item
    - rem = 1000*item: score ≈ -2*item
    
    So the score is highest (closest to 0) when rem is small, and lowest (most negative)
    when rem is large. This is exactly best-fit behavior!
    
    But wait, let's check the derivative. Let f(rem) = 2*rem*item/(rem+item).
    f'(rem) = 2*item * (rem+item - rem)/(rem+item)^2 = 2*item^2/(rem+item)^2 > 0
    
    So f(rem) is increasing in rem, meaning -f(rem) is decreasing in rem.
    So we prefer smaller rem. This is best-fit.
    
    However, the harmonic mean has a nice property: it's a smooth function that
    penalizes both very small and very large values of rem relative to item.
    But since we're taking the negative, and we want to maximize the score,
    we prefer bins where the harmonic mean is small, i.e., where rem is small.
    
    Let's also consider a variation: what if we use the arithmetic mean instead?
    A(rem, item) = (rem + item)/2
    score = -A = -(rem+item)/2 = -rem/2 - item/2
    This is linear in rem, so it's equivalent to best-fit with a linear penalty.
    
    The harmonic mean is more sensitive to small rem values. For example:
    - item = 10
    - rem = 1: H = 2*1*10/11 ≈ 1.82, score ≈ -1.82
    - rem = 10: H = 2*10*10/20 = 10, score = -10
    - rem = 100: H = 2*100*10/110 ≈ 18.18, score ≈ -18.18
    
    So the penalty increases as rem increases, but with diminishing sensitivity
    for large rem (since H approaches 2*item as rem → ∞).
    
    This seems like a good smooth surrogate for best-fit. Let's implement it.
    
    We also need to handle the case where rem = 0. In that case, H = 0, score = 0.
    This is fine.
    
    Let's also consider adding a small epsilon to avoid division by zero, though
    rem + item > 0 since item > 0 and rem >= 0.
    """
    t0 = time.time()
    
    # Bins that can fit the item
    remaining = bins - item
    valid = remaining >= 0
    
    # Compute scores using harmonic mean surrogate
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    if np.any(valid):
        rem = remaining[valid]
        # Harmonic mean: H = 2 * rem * item / (rem + item)
        # Add small epsilon to avoid any numerical issues
        denom = rem + item
        # Handle case where denom might be zero (shouldn't happen since item > 0)
        denom = np.maximum(denom, 1e-9)
        H = 2.0 * rem * item / denom
        scores[valid] = -H
    
    # Deadline guard
    if time.time() - t0 > 110:
        # Fallback: pure best-fit
        scores = np.where(valid, -remaining, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

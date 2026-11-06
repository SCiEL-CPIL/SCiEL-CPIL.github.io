"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Strategy: Smooth surrogate for best-fit-decreasing with fragmentation awareness.
    
    The L1 lower bound is sum(items)/capacity. To minimize excess over L1,
    we want to minimize the number of bins used, which means filling bins as
    completely as possible. Best-fit (minimize remaining) is the classic heuristic.
    
    However, pure best-fit can create tiny fragments that waste capacity.
    A smooth surrogate that penalizes remaining capacity with a slightly convex
    shape can help: it strongly prefers small remainders but with a soft knee
    that avoids over-penalizing moderate remainders (which might still fit future items).
    
    We use: score = -remaining^alpha  where alpha < 1
    For alpha < 1, the function is concave in remaining, meaning:
    - Small differences in small remainders matter a lot (strong preference for near-perfect fit)
    - Large differences in large remainders matter less (flexibility for future items)
    
    This is equivalent to -sqrt(rem) when alpha=0.5, which is what the seed uses.
    Let's try alpha=0.7 or alpha=0.8 for a slightly different curvature.
    
    Actually, let's think about this more carefully. The key insight from the literature
    is that for online bin packing, best-fit-decreasing (when items are sorted) performs well,
    but online we don't have that luxury. Best-fit is a strong baseline.
    
    Let's try a different approach: instead of just minimizing remaining, we can
    add a small bonus for bins that have a "nice" remaining capacity that is likely
    to fit future items. For Weibull distributions, items tend to be smaller, so
    having a remaining capacity that is a multiple of a typical item size might be good.
    
    But that's too complex. Let's stick with a smooth best-fit variant.
    
    Let's try: score = -(remaining + epsilon)^alpha
    with alpha = 0.75 (between sqrt and linear)
    
    Or, we can use a log-sum-exp style soft-min to create a smooth approximation
    of the min operation, but that's overkill for a single item placement.
    
    Let's go with alpha = 0.75 and see if it improves over the seed's alpha = 0.5.
    """
    t0 = time.time()
    
    # Bins that can fit the item
    remaining = bins - item
    valid = remaining >= 0
    
    # Compute scores
    # Smooth surrogate: -(remaining + epsilon)^alpha
    # alpha < 1 gives a concave penalty, strongly preferring small remainders
    # but with diminishing sensitivity for larger remainders
    alpha = 0.75
    epsilon = 1e-9
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    if np.any(valid):
        rem = remaining[valid]
        scores[valid] = -(rem + epsilon) ** alpha
    
    # Deadline guard
    if time.time() - t0 > 110:
        # Return fallback: pure best-fit
        scores = np.where(valid, -remaining, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

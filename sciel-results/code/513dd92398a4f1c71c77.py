import numpy as np
import time

# Keep the same entry-point name and return type.
_INFEASIBLE = -1e18

# Tunable hyperparameters for the scoring heuristic.
# The idea: prefer bins that, after placing the item, have the least
# remaining capacity (tightest fit), but also penalize "wasteful"
# placements where the remaining capacity is too large relative to the
# bin's original capacity, encouraging consolidation into already-used
# bins to reduce the total number of bins opened.

# We use a smooth surrogate for the "waste" concept.
# For each feasible bin, after placing item, remaining = bins[i] - item.
# We want to minimize remaining, but also avoid creating bins with very
# large remaining capacity (which will likely be wasted).

# Strategy: score = -remaining + alpha * (bins[i] / capacity_proxy)
# But we don't know the original capacity directly. However, bins[i] is
# the remaining capacity. A bin with high remaining relative to typical
# item sizes is less desirable for small items because it "wastes" space.

# Alternative: Use the concept of "fit ratio" = item / bins[i] (how well
# the item fits into the bin). Higher fit ratio means the item uses a
# larger fraction of the bin's remaining capacity, which is generally
# better for packing density.

# Let's try: score = fit_ratio - beta * remaining
# where fit_ratio = item / bins[i] for feasible bins.
# This encourages placing items in bins where they take up a larger
# fraction of the remaining space.

# But for very small bins, fit_ratio can be very large, which might
# cause issues. Let's use a bounded version.

# Actually, let's think about this differently. The L1 lower bound is
# ceil(sum(items) / capacity). To minimize the number of bins used,
# we want to maximize the fill ratio of each bin. 

# A good heuristic: prefer the bin where, after placing the item, the
# remaining capacity is smallest (tightest fit), but break ties by
# preferring bins that are already more filled (higher original usage).

# Since we only have remaining capacity, a bin with small remaining
# capacity is already well-filled. So tightest fit (minimize remaining)
# already implicitly favors consolidation.

# Let's try a different angle: use the square of the remaining capacity
# to penalize large remaining gaps more heavily.
# score = -remaining^2 / (item + epsilon)
# This makes the penalty for wasted space quadratic, which might lead
# to better packing.

# Or even simpler: score = -remaining / item
# This normalizes the remaining capacity by the item size, so it's
# scale-invariant. A bin with remaining capacity of 10 for an item of
# size 5 has a score of -2, while a bin with remaining capacity of 20
# for the same item has a score of -4. We prefer the first.

# Let's try: score = -remaining / (item + 1e-9)
# This is a simple, scale-invariant tightest-fit heuristic.

# But wait, the current seed uses -remaining + beta * bins. Let's try
# something different: a smooth maximum-like function.

# Idea: Use log-sum-exp to create a smooth approximation of the minimum
# remaining capacity, and use that as a baseline. But this is overkill
# for a per-item decision.

# Let's try a simple improvement: instead of linear -remaining, use
# -remaining^2 to penalize large gaps more.
# score = -remaining^2 / (capacity_estimate + epsilon)
# But we don't know capacity. Let's just use -remaining^2.

# Actually, let's test multiple strategies and pick the best one based
# on intuition. I'll go with a combination:
# 1. Primary: -remaining (tightest fit)
# 2. Secondary: -remaining^2 (quadratic penalty for waste)
# 3. Tertiary: prefer bins with higher original fill (approximated by
#    higher bins[i] before placement, but we only have remaining).

# Let's try: score = -remaining - gamma * remaining^2
# with gamma > 0 to penalize large gaps quadratically.

_GAMMA = 0.01


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each bin with enough remaining capacity; higher = preferred.
    
    Uses a quadratic penalty on remaining capacity to discourage
    placements that leave large gaps, which tends to reduce the total
    number of bins needed compared to pure tightest-fit.
    """
    bins = np.asarray(bins, dtype=float)
    remaining = bins - item
    feasible = remaining >= 0.0

    # Quadratic penalty: prefer bins where remaining is small.
    # score = -remaining - gamma * remaining^2
    # This is more aggressive than linear in penalizing large gaps.
    score = -remaining - _GAMMA * remaining * remaining

    # Mask infeasible bins.
    score = np.where(feasible, score, _INFEASIBLE)

    return score

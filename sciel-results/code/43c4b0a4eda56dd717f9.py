import numpy as np
import time

# Keep the same entry-point name and return type.
_INFEASIBLE = -1e18

# Tunable hyperparameters for the forward-looking utility heuristic.
# The idea: Instead of just minimizing the immediate waste (remaining capacity),
# we estimate the "utility" of the remaining capacity for future items.
#
# We assume items follow a Weibull distribution. For capacity 100, typical
# online bin packing benchmarks often use shapes that result in a mean
# around 20-30. Let's estimate the mean item size.
# If we don't know the exact distribution parameters, we can use a robust
# heuristic: the mean item size is roughly 1/4 to 1/5 of the capacity
# for many standard distributions in this context.
# Let's set EXPECTED_NEXT_ITEM to a value that represents a "typical"
# next item. If the remaining capacity after placing the current item is
# significantly larger than this, it's "wasted" because it's too big for
# most future items (which will be smaller on average, but the tail might
# not fit). If it's smaller, it's unusable for the expected next item.
#
# Actually, a better metric is: How much capacity remains that is
# "usable" for the next item?
# If remaining > expected_next, the bin is still "alive" for the next item.
# If remaining < expected_next, the bin might be "dead" for the next item
# (unless the next item is very small).
#
# Let's define a utility function U(remaining, expected_next):
# We want to maximize the probability that the bin can accept the next item.
# Or simply, we want to keep the remaining capacity in a "sweet spot".
#
# Strategy:
# 1. Calculate remaining capacity for each feasible bin.
# 2. Estimate the mean next item size (mu).
# 3. Score = - (distance of remaining capacity from an ideal target).
#    What is the ideal target?
#    If we place the item, we want the remaining space to be just enough
#    for one more typical item, or zero.
#    If remaining is very large, we've wasted space (could have used a smaller bin).
#    If remaining is very small (but > 0), it's good (tight fit).
#    But if remaining is slightly above mu, it's also good because it
#    guarantees the next item fits.
#
# Let's use a penalty based on "inefficiency".
# Inefficiency = max(0, remaining - mu) + max(0, mu - remaining) * penalty_factor?
# No, that's just absolute distance.
#
# Let's try: Score = - (remaining^2 / (2 * mu)) if remaining < mu
#              Score = - (mu + (remaining - mu)^2 / (2 * mu)) if remaining >= mu
# This creates a V-shape centered at mu, but asymmetric.
# Actually, simpler: We want to minimize the "dead space".
# Dead space is capacity that cannot be used by the next item.
# If remaining < mu, dead space is 0 (we hope the next item is smaller).
# If remaining > mu, dead space is (remaining - mu) approximately?
#
# Let's refine:
# The goal is to minimize the number of bins.
# This is equivalent to maximizing the fill rate.
# A bin is "full" when it can't take another item.
# We want to fill bins as much as possible before opening new ones.
#
# Heuristic:
# Prefer bins where the remaining capacity is closest to the expected next item size.
# Why? Because if remaining == expected_next, the bin will likely be full after the next item.
# If remaining >> expected_next, we have wasted space.
# If remaining << expected_next, the bin might be full already (for the next item),
# which is also good, but we might have under-utilized the current item's placement
# if a slightly larger bin would have allowed better future packing.
#
# Actually, Best Fit (minimize remaining) is already close to this.
# The improvement comes from recognizing that "too tight" isn't always bad,
# but "too loose" is definitely bad.
#
# Let's use a score that penalizes large remaining capacity more heavily
# than small remaining capacity.
# Score = - (remaining / mu)^alpha
# With alpha > 1, this penalizes large gaps more.
#
# Let's try alpha = 1.5 or 2.0.
# Also, we need to handle the case where remaining is very small.
#
# Let's implement:
# mu = 25.0 (Estimated mean item size for capacity 100)
# score = - (remaining / mu)^1.5

_MU = 25.0
_ALPHA = 1.5


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each bin with enough remaining capacity; higher = preferred.
    
    Uses a forward-looking utility heuristic. It estimates the expected size
    of the next item and penalizes bins that leave a remaining capacity
    significantly larger than this estimate, as this represents wasted space
    that likely cannot be filled by the next item.
    """
    bins = np.asarray(bins, dtype=float)
    remaining = bins - item
    feasible = remaining >= 0.0

    # Calculate the base score based on remaining capacity.
    # We want to minimize the "wasted" potential.
    # If remaining is small, it's good (tight fit).
    # If remaining is large, it's bad (waste).
    # We use a power law to emphasize the penalty for large gaps.
    
    # Avoid division by zero or negative base for power if remaining is 0.
    # remaining is >= 0 for feasible bins.
    safe_remaining = np.maximum(remaining, 1e-9)
    
    # Normalize by expected next item size
    normalized_remaining = safe_remaining / _MU
    
    # Apply power penalty
    penalty = np.power(normalized_remaining, _ALPHA)
    
    # Score is negative penalty (higher is better, so we want lower penalty)
    score = -penalty
    
    # Mask infeasible bins
    score = np.where(feasible, score, _INFEASIBLE)
    
    return score

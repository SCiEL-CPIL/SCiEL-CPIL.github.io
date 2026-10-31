# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 2.0

# Weibull parameters for the item distribution (shape k, scale lambda)
# Standard bin packing benchmarks often use k=2, lambda=10 for capacity 100
# This yields mean item size ~8.86, which is reasonable for 1000 items in ~113 bins
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 10.0

# Precompute the Weibull CDF for remainders
# CDF(x) = 1 - exp(-(x/lambda)^k)
# We want to score bins by the "regret" of the remainder they create.
# A remainder r is "bad" if it's likely to remain unused (wasted).
# This happens if no future item fits into it, or if the items that fit are small and leave a new small remainder.
# A simpler proxy: the probability that a future item is LARGER than the remainder.
# If a future item is larger than r, it can't fit, so r stays as is.
# But if many items are smaller, they might fill it up.
# Let's use the survival function S(r) = exp(-(r/lambda)^k).
# High S(r) means it's likely that a random future item is > r, so r is "stuck".
# Actually, we want to MINIMIZE the chance that the remainder is wasted.
# If we create a remainder r, the "regret" is high if r is large and hard to fill, or if r is small and unusable.
#
# Let's think about "perfect fit".
# If the remainder is 0, it's perfect.
# If the remainder is r, the probability that the NEXT item fits is 1 - S(r) = CDF(r).
# If the next item fits, the new remainder is r - item.
# This is a recursive process.
#
# Approximation:
# The "value" of a remainder r is the expected number of items that can be packed into it before it becomes "unusable" (e.g., remainder < min_item_size or just small).
# Or simpler: The probability that the remainder r is eventually filled completely (or close to it) is low for most r.
#
# Let's use a different angle:
# Score = - (Probability that remainder r is "bad")
# What is a "bad" remainder?
# 1. Very small remainders (< 5) are often unusable for future items if the min item size is larger. But Weibull has support [0, inf), so small items exist.
# 2. Large remainders are "bad" in the sense that they represent wasted capacity if not filled.
#
# The directive suggests: "score as the negative of the cumulative distribution function"
# Wait, the directive says: "calculate the score as the negative of the cumulative distribution function"
# Which CDF? "probability that this remainder cannot be perfectly filled"
# This is vague. Let's interpret "cannot be perfectly filled" as "the remainder is left over at the end".
#
# Let's try a specific model:
# The "regret" of creating a remainder r is proportional to the probability that r is larger than the typical remaining items.
# Or, more simply, let's use the CDF of the item distribution itself.
# If we place an item in a bin, the new remainder is r = bin_cap - item.
# We want to choose the bin that minimizes the "regret" of r.
#
# Hypothesis: Remainders that are close to the "median" or "mean" of the item distribution are "good" because they are likely to be filled by a future item.
# Remainders that are very small or very large are "bad".
#
# Let's define a "fitness" function for a remainder r:
# Fitness(r) = CDF(r) * (1 - CDF(r))? No.
#
# Let's look at the inspiration: "score as the negative of the cumulative distribution function"
# Maybe: Score = -CDF(r)?
# If r is small, CDF(r) is small, so -CDF(r) is close to 0.
# If r is large, CDF(r) is close to 1, so -CDF(r) is close to -1.
# So we would prefer small remainders? That's Best Fit.
#
# Let's try: Score = -S(r) = -(1 - CDF(r)) = CDF(r) - 1.
# If r is small, CDF(r) is small, score is negative.
# If r is large, CDF(r) is large, score is close to 0.
# So we would prefer LARGE remainders? That's Worst Fit.
#
# Let's try a combination.
# The "regret" of a remainder r is the probability that it remains unfilled.
# This is roughly the probability that no future item fits into it.
# But items arrive sequentially.
#
# Let's use the expected "waste".
# Waste = r * P(r is not filled).
#
# Let's try a structural change:
# Instead of a fixed heuristic, we use the Weibull distribution to estimate the "utility" of a remainder.
# Utility(r) = Expected number of future items that can fit into r.
# This is an infinite sum, but we can approximate it.
#
# Simpler structural change:
# Use the derivative of the CDF (the PDF) to find "sweet spots" for remainders.
# If r is near the mode of the distribution, it's a "good" remainder.
# Mode of Weibull(k, lambda) = lambda * ((k-1)/k)^(1/k) for k > 1.
# For k=2, mode = lambda * 0.5 = 5.
# So remainders around 5 are "good".
#
# Let's score based on the distance from the mode.
# Score = -|r - mode|?
#
# Let's try the directive's suggestion more literally.
# "score each bin based on the expected future regret of the remainder it creates, approximated by the probability that this remainder cannot be perfectly filled by items from the known Weibull distribution."
#
# "Cannot be perfectly filled" might mean "the remainder is not 0".
# The probability that a remainder r is NOT 0 is 1. That's not helpful.
#
# Maybe "perfectly filled" means "filled by a single item".
# Probability that a single item fits exactly into r is 0 (continuous).
# Probability that a single item fits into r is CDF(r).
# So the probability that it CANNOT be filled by a single item is 1 - CDF(r) = S(r).
# So Regret(r) = S(r).
# Score = -S(r) = CDF(r) - 1.
# This prefers large remainders (Worst Fit).
#
# What if "perfectly filled" means "filled by any sequence of items"?
# This is the "bin packing problem" itself.
#
# Let's try a hybrid:
# Score = - (S(r) * r)
# This penalizes large remainders that are likely to be "stuck" (high S(r)) and large in size (high r).
#
# Let's implement this and see.
# We also need to handle the case where r is very small.
# If r is very small, S(r) is small, so -S(r)*r is close to 0.
# But small remainders are also bad because they are hard to fill.
#
# Let's try: Score = - (S(r) * r + CDF(r) * (1/r) * 100) ? No, that's getting complicated.
#
# Let's stick to the simplest interpretation of the directive:
# Score = -CDF(r)
# This is Best Fit.
#
# Score = -S(r)
# This is Worst Fit.
#
# Let's try a new direction:
# The "regret" of a remainder r is the probability that r is larger than the next item.
# This is S(r).
# If S(r) is high, it's likely that the next item won't fit, so r stays.
# If r stays, it's "wasted" capacity.
# So we want to minimize S(r) * r.
#
# Let's implement: Score = - (S(r) * r)
# This is a structural change from the linear combination in the seed.
# It uses the Weibull distribution explicitly.

def _weibull_cdf(x, k=_WEIBULL_K, lam=_WEIBULL_LAMBDA):
    """Calculate the CDF of a Weibull distribution."""
    if x <= 0:
        return 0.0
    return 1.0 - np.exp(-((x / lam) ** k))

def _weibull_sf(x, k=_WEIBULL_K, lam=_WEIBULL_LAMBDA):
    """Calculate the Survival Function (1 - CDF) of a Weibull distribution."""
    if x <= 0:
        return 1.0
    return np.exp(-((x / lam) ** k))

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy:
    - Only consider bins with remaining capacity >= item.
    - Score based on the "regret" of the remainder created.
    - Regret is approximated by S(r) * r, where r is the new remainder.
    - S(r) is the probability that a future item is larger than r.
    - This penalizes large remainders that are likely to be "stuck".
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        # Fallback to simple Best Fit if we are running out of time
        scores = np.full(len(bins), -1e18)
        remaining = bins - item
        valid_mask = remaining >= 0
        if np.any(valid_mask):
            scores[valid_mask] = -remaining[valid_mask]
        return scores

    scores = np.full(len(bins), -1e18)
    
    # Calculate remaining capacity in each bin
    remaining = bins - item
    
    # Only consider bins where remaining >= 0
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_indices]
    
    # Calculate the regret for each valid remainder
    # Regret = S(r) * r
    # We want to minimize regret, so Score = -Regret
    sf_values = _weibull_sf(valid_remaining)
    regret = sf_values * valid_remaining
    scores[valid_indices] = -regret
    
    return scores
# EVOLVE-BLOCK-END

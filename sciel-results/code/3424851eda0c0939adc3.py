"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-computed constants for Weibull(1, 1) item distribution
# The task uses Weibull sequences. Weibull(k=1, scale=1) is exponential.
# Common in bin packing benchmarks.
# Mean of Weibull(k, scale) = scale * gamma(1 + 1/k)
# For k=1, scale=1: mean = 1
# For k=1, scale=1: variance = 1
# Let's assume the standard Weibull(1, 1) or similar.
# The prompt mentions "five generated Weibull sequences".
# Typical parameters for such tasks: Weibull(k=1, scale=1) or Weibull(k=1.5, scale=...).
# Let's look at the "Expected Dead Space" clue: r * S(r).
# S(r) is the survival function.
# For Weibull(k, scale), S(r) = exp(-(r/scale)^k).
# If k=1, S(r) = exp(-r/scale).
# Let's assume Weibull(1, 1) as a baseline, but we can adapt.
# However, the function signature is fixed: priority(item, bins).
# We don't get the distribution parameters.
# But the "Knowledge" says: "Replaces heuristic 'dead zone' estimation with a distribution-aware 'Expected Dead Space' metric (r * S(r))... Measured: 0.8196 -> 1.817."
# This suggests that using r * S(r) as a penalty for the remaining space is effective.
# The score is "mean percent excess of bins used over the L1 lower bound".
# Lower is better.
# L1 lower bound is sum(items) / capacity.
# Excess = (bins_used - LB) / LB.
# We want to minimize bins_used.
# This is equivalent to minimizing the total wasted space, or maximizing the fill ratio.
# Best-fit minimizes the remaining space immediately, but might create small unusable gaps.
# First-fit decreasing is good offline, but online we don't know future items.
# The "Expected Dead Space" idea:
# If a bin has remaining space r, the expected amount of that space that will be wasted
# (i.e., not filled by future items) depends on the distribution of future items.
# If future items are Weibull, the probability that an item of size x fits in r is 1 for x<=r, 0 for x>r.
# But items arrive sequentially.
# A common heuristic is to avoid leaving remainders that are too small to be filled by typical items.
# The metric r * S(r) where S(r) is the survival function of the item size distribution?
# No, S(r) usually means P(X > r).
# If we leave remainder r, the "dead space" is the part of r that cannot be filled.
# If items are i.i.d. Weibull(k, scale), the expected number of items that fit in r is... complex.
# Let's look at the specific formula mentioned: r * S(r).
# If S(r) = P(X > r), then for large r, S(r) is small.
# For Weibull(1, 1), S(r) = exp(-r).
# So the penalty is r * exp(-r).
# This function peaks at r=1 with value 1/e.
# It is small for r -> 0 and r -> infinity.
# This suggests we should avoid remainders around 1?
# Or is it a cost function for the remainder?
# If we use this as a score to minimize, we prefer bins where r * S(r) is small.
# This means we prefer r close to 0 or r very large.
# But r cannot be very large if the bin capacity is 100 and items are ~1.
# Wait, if r is large, it means the bin is almost empty.
# If we prefer almost empty bins, that's First-Fit behavior.
# If we prefer r close to 0, that's Best-Fit behavior.
# The function r * exp(-r) is small at r=0 and large at r=1.
# So it penalizes remainders around 1.
# This makes sense if items are around size 1. A remainder of 1 is "dead" because the next item (size ~1) might not fit or will leave another small remainder.
# Actually, if item size is ~1, a remainder of 1 can fit an item of size 1.
# But if the item is slightly larger than 1, it won't fit.
# So remainders in the range [0, mean_item_size] are risky.

# Let's assume Weibull(k=1, scale=1) for the S(r) calculation.
# S(r) = exp(-r).
# Penalty = r * exp(-r).
# We want to minimize this penalty.
# So we want to choose the bin with the minimum penalty among feasible bins.
# Feasible bins have bins[i] >= item.
# Remaining space r = bins[i] - item.
# Penalty p = r * exp(-r).
# We return -p so that the highest score (least negative) is the best (smallest penalty).
# Or we can return -penalty.

# Let's refine the distribution.
# The prompt says "Weibull sequences".
# If the parameters are different, the S(r) changes.
# Weibull(k, scale) PDF: f(x) = (k/scale) * (x/scale)^(k-1) * exp(-(x/scale)^k)
# CDF: F(x) = 1 - exp(-(x/scale)^k)
# Survival: S(x) = exp(-(x/scale)^k)
# If k=1, scale=1, S(x) = exp(-x).
# If k=1.5, scale=1, S(x) = exp(-x^1.5).
# Without knowing the parameters, we have to guess.
# However, the "Knowledge" explicitly mentions "Expected Dead Space" metric (r * S(r)).
# And it improved the score significantly.
# So let's use this form.
# We need to choose k and scale.
# Common default is k=1, scale=1.
# Let's try that.

# To be safe, we can check the time and return a fallback if needed.
# But this calculation is O(N) which is fast.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify feasible bins
    feasible = remaining >= 0
    
    # Initialize scores with -infinity for infeasible bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # If no feasible bins, return scores (all -inf)
    # The caller should handle this (e.g., open new bin), but the contract says
    # "scores each bin with enough remaining capacity".
    # If none, we return all -inf.
    
    if not np.any(feasible):
        return scores
    
    # Calculate penalty for feasible bins
    # Penalty = r * S(r)
    # Assume Weibull(1, 1) -> S(r) = exp(-r)
    r = remaining[feasible]
    # Avoid numerical issues if r is very large (exp(-r) -> 0)
    # But r <= 100, so exp(-100) is tiny, fine.
    s_r = np.exp(-r)
    penalty = r * s_r
    
    # We want to minimize penalty, so score = -penalty
    # Higher score is better.
    scores[feasible] = -penalty
    
    return scores
# EVOLVE-BLOCK-END

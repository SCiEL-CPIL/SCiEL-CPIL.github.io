"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precomputed constants for speed
_CAPACITY = 100.0
# Weibull shape parameter guess (common in packing benchmarks)
_SHAPE = 1.5
# Scale parameter for Weibull (typical item size ~ 20-30 for cap 100)
# If mean item is ~25, scale = 25 / Gamma(1 + 1/1.5) ≈ 25 / 0.91 ≈ 27.5
_SCALE = 27.0

# Precompute a lookup table for the Weibull CDF to speed up scoring
# We want to score based on how "good" the remainder is.
# A good remainder is one that is likely to be filled by future items.
# Actually, the previous insight was: "penalize large remainders more heavily... favoring remainders that are statistically likely to be filled".
# Wait, if we use Best-Fit, we minimize remainder.
# The improved program (1.031) used Weibull CDF.
# Let's try a score that combines Best-Fit with a statistical adjustment.
# Score = -remainder - alpha * P(future item fits in remainder) ?
# Or maybe Score = -remainder * (1 + beta * CDF(remainder))?

# Let's look at the "golden ratio" strategy mentioned in the directive.
# "Model the bin state as a 1D spatial coordinate where the score is the negative Euclidean distance between the resulting remainder and a 'golden ratio' target"
# Target = 0.618 * capacity = 61.8
# Score = -|remainder - 61.8|
# This encourages remainders to be around 61.8, which is quite large. This seems counter-intuitive for Best-Fit, but might reduce fragmentation by keeping bins "medium" full?
# Actually, if remainder is 61.8, used is 38.2. That's low fill.
# Maybe the target is for the *used* space? No, "resulting remainder".
# If we target remainder ~ 61.8, we are leaving a lot of space.
# Let's stick to the Weibull insight which measured 1.031.

# Strategy:
# 1. Filter bins with enough capacity.
# 2. Calculate remainder = bin - item.
# 3. Score based on remainder.
#    - Linear Best Fit: score = -remainder
#    - Weibull-adjusted: The idea is that small remainders are "wasted" if they can't fit future items.
#      If we assume future items follow Weibull(shape, scale), the probability that a future item fits in remainder R is CDF(R).
#      If we leave a remainder R, its "value" is proportional to the expected waste.
#      Actually, a common heuristic is "Worst Fit Decreasing" or "Best Fit".
#      Let's try a score that prefers remainders that are "useful".
#      Useful remainder: one that has a high probability of being filled by a future item?
#      If P(future item <= R) is high, then R is likely to be filled, so it's "good" to leave?
#      No, if R is likely to be filled, it's good. If R is unlikely to be filled (too small for most items), it's bad (fragmentation).
#      So we want to maximize P(future item <= R) subject to minimizing R?
#      Or minimize R, but penalize R that is "too small" (fragmentation).
#      Let's try: Score = -R + lambda * CDF(R)
#      This rewards smaller R (Best Fit) but also rewards R that are statistically likely to be filled.
#      Wait, CDF(R) increases with R. So this term penalizes small R (if R is small, CDF is small).
#      This creates a trade-off: don't make R too small (fragmentation), but don't make R too large (waste).
#      This creates a local minimum for R.

# Let's precompute CDF for remainders from 0 to 100 with step 0.1
# R_max = 100
# Step = 0.1 -> 1000 points
_R_MAX = 100.0
_STEP = 0.1
_R_GRID = np.arange(0, _R_MAX + _STEP, _STEP)
# Weibull CDF: 1 - exp(-(x/scale)^shape)
_CDF_GRID = 1.0 - np.exp(-((_R_GRID / _SCALE) ** _SHAPE))

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Returns an array of scores. Higher score is better.
    """
    # 1. Filter bins
    # bins is an array of remaining capacities? Or total capacities?
    # "bins: np.ndarray" usually contains remaining capacities in this context,
    # or the evaluator passes remaining capacities.
    # The seed program: return -(bins - item). This implies `bins` contains remaining capacity.
    # If `bins` was total capacity, we'd need to subtract used. But seed uses `bins - item` as remainder.
    # So `bins` = remaining capacity.
    
    valid = bins >= item
    if not np.any(valid):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # Get valid remainders
    remainders = bins[valid] - item
    
    # 2. Calculate Scores
    # Interpolate CDF for the remainders
    # Clamp remainders to grid range just in case
    r_clamped = np.clip(remainders, 0, _R_MAX)
    # Linear interpolation for CDF
    cdf_vals = np.interp(r_clamped, _R_GRID, _CDF_GRID)
    
    # Score = -R + lambda * CDF(R)
    # Let's tune lambda. 
    # If lambda is large, we prioritize "fillable" remainders (larger R).
    # If lambda is small, we prioritize Best Fit (smaller R).
    # The previous improvement used a Weibull CDF-based score.
    # Let's try lambda = 5.0 (arbitrary start, scaled to remainder magnitude ~20-80)
    # R is in [0, 100]. CDF is in [0, 1].
    # To balance, lambda should be comparable to R.
    # Let's try lambda = 10.0
    
    LAMBDA = 10.0
    scores = -remainders + LAMBDA * cdf_vals
    
    # 3. Construct output array
    # Initialize with -inf for invalid bins
    out_scores = np.full_like(bins, -np.inf, dtype=float)
    out_scores[valid] = scores
    
    return out_scores

# EVOLVE-BLOCK-END

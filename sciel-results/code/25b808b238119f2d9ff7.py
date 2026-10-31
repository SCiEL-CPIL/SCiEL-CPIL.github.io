"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull parameters for the generated sequences
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 15.0

# Adversarial robustness parameters
# We model the "regret" as the maximum item size that could fit in the remainder
# but is unlikely to arrive soon, or the wasted space if a larger item arrives.
# We use a quantile-based approach: for a given remainder r, what is the
# "worst-case" useful item size?
# Actually, the directive suggests: "negative of the maximum possible future regret,
# defined as the size of the largest Weibull-distributed item that could fit into
# the resulting remainder".
# This is tricky. If r is large, the largest item that fits is r itself.
# But if r is small, the largest item that fits is small.
# The "regret" is usually associated with leaving space that *cannot* be used.
# Let's reinterpret:
# If we leave remainder r, we lose the opportunity to place any item of size > r.
# The "regret" of choosing a bin with remainder r is the expected size of items
# that we *cannot* place in that remainder, weighted by their probability of arrival?
# Or, more simply, following the "adversarial" hint:
# Assume the next item is drawn from the Weibull distribution.
# If the next item x <= r, it fits. Regret = 0 (or small).
# If the next item x > r, it does NOT fit. We must open a new bin or use another bin.
# The "cost" of x > r is that we failed to utilize this bin's capacity for x.
# However, the directive says: "size of the largest Weibull-distributed item that
# could fit into the resulting remainder".
# This sounds like we want to maximize the capacity for large items?
# No, "prioritizing bins that create remainders robust against the worst-case
# immediate future demand".
# If the future demand is a large item, we want a large remainder.
# If the future demand is a small item, we want a small remainder (to avoid waste).
# "Worst-case" usually implies maximizing the minimum utility or minimizing the maximum loss.
#
# Let's try a different structural interpretation:
# Regret(r) = E[ max(0, X - r) ] where X ~ Weibull.
# This is the expected "shortfall" or "wasted potential" if the next item X is larger than r.
# If X <= r, we can place it, and the "regret" of not having more space is 0?
# Actually, if X <= r, we place it and the remaining space becomes r - X.
# The "regret" of the *decision* to pick this bin is hard to define locally.
#
# Let's look at the "largest item that could fit" phrasing again.
# Max item that fits in r is r itself.
# This metric is just r. Maximizing r is Worst Fit.
# But the score is "negative of the maximum possible future regret".
# If Regret = max_item_fit, then Score = -r. This is Best Fit.
# That's circular.
#
# Let's try: Regret(r) = E[ X * 1[X > r] ] + r * P[X <= r] ?
# No.
#
# Let's go with a robust optimization metric:
# Minimize the expected "wasted space" in the next step.
# If we pick bin with remainder r:
# 1. If next item X <= r: It fits. New remainder is r - X. The "waste" is effectively 0 for this step (item placed).
# 2. If next item X > r: It does not fit. We must use another bin. The "waste" is that we didn't use this bin.
#
# A common robust metric in packing is the "Expected Fragmentation".
# Let's use the Expected Shortfall of the remainder relative to the distribution of items.
# Specifically, we want remainders that are either very small (tight fit) or very large (flexible).
# The "Adversarial" view: The adversary chooses the next item to maximize our regret.
# If we have remainder r, the adversary picks X.
# If X <= r, we place it. Regret = 0?
# If X > r, we can't place it. Regret = X? (We lost the chance to place X here, so we have to put it elsewhere, potentially opening a new bin).
# To be robust against the adversary, we should minimize the maximum regret.
# Max regret for remainder r is achieved when X is just slightly larger than r?
# Or when X is huge?
# If X is huge, we can't place it anyway, so the regret is the same regardless of r (as long as r < X).
# The "regret" is really about the *opportunity cost*.
#
# Let's try a specific structural change:
# Score = - ( w1 * DeadSpace(r) + w2 * E[ (X - r)+ ] )
# Where E[ (X - r)+ ] is the expected part of X that exceeds r.
# This penalizes remainders that are likely to be too small for the next item.
# If r is small, E[(X-r)+] is large (many items will exceed r).
# If r is large, E[(X-r)+] is small (few items will exceed r).
# This encourages larger remainders (Worst Fit bias) to accommodate large items.
# But we also have DeadSpace which penalizes medium remainders.
#
# Let's compute E[ (X - r)+ ] for Weibull.
# E[ (X - r)+ ] = ∫_r^∞ (x - r) f(x) dx
# = ∫_r^∞ x f(x) dx - r * ∫_r^∞ f(x) dx
# = ∫_r^∞ x f(x) dx - r * S(r)
#
# For Weibull(k, lambda):
# f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)
# S(r) = exp(-(r/lambda)^k)
#
# This integral doesn't have a simple closed form for general k, but we can approximate it or use scipy.
# However, we need speed.
#
# Alternative: Use the Mean Residual Life function.
# E[X | X > r] = (∫_r^∞ x f(x) dx) / S(r)
# So E[(X-r)+] = S(r) * (E[X | X > r] - r)
#
# Let's implement this. It captures the "robustness" against large items.
# If r is small, S(r) is high, and E[X|X>r] is high, so the penalty is high.
# If r is large, S(r) is low, so the penalty is low.
# This favors large remainders.
# Combined with DeadSpace (which penalizes medium remainders), we might get a balance.
#
# Let's refine the combination:
# Penalty(r) = DeadSpace(r) + Alpha * E[(X-r)+]
#
# We need to compute E[(X-r)+] efficiently.
# We can precompute a lookup table or use an approximation.
# Since r is continuous, a lookup table with interpolation is good.
# Or, we can use the fact that for Weibull, the mean excess loss is smooth.
#
# Let's use scipy.stats.weibull_min to compute the survival function and the mean excess.
# Actually, scipy.stats.weibull_min has a method `sf` for survival function.
# For the integral, we might need to do numerical integration or use a precomputed table.
# Given the 150s limit, precomputation is safe.
#
# Let's precompute E[(X-r)+] for r in [0, 100] with step 0.1.
#
# Wait, the current program uses a KDE approach. The inspiration 1 uses a simpler dead space.
# The directive says: "Treat the bin packing process as a game-theoretic adversarial robustness problem...".
#
# Let's stick to the precomputed table for speed and accuracy.

# Precompute the expected shortfall E[(X - r)+] for r in [0, 100]
_R_MAX = 100.0
_R_STEP = 0.1
_R_GRID = np.arange(0, _R_MAX + _R_STEP, _R_STEP)
_WEIBULL_DIST = scipy.stats.weibull_min(c=_WEIBULL_K, scale=_WEIBULL_LAMBDA)

# Calculate S(r) and E[X | X > r]
# S(r) = sf(r)
# E[X | X > r] = (∫_r^∞ x f(x) dx) / S(r)
# Note: ∫_r^∞ x f(x) dx can be computed as -d/dr ∫_r^∞ F(x) dx ? No.
# Let's just use numerical integration on the grid or a recursive relation?
# Actually, for Weibull, we can use the property:
# ∫_r^∞ x f(x) dx = ∫_0^∞ (r+t) f(r+t) dt = r S(r) + ∫_0^∞ t f(r+t) dt
# This doesn't help much.
#
# Let's use scipy.integrate.quad for the precomputation. It's fast enough for 1000 points.
from scipy.integrate import quad

def _expected_shortfall(r):
    """Compute E[(X - r)+] for Weibull distribution."""
    if r >= _R_MAX:
        return 0.0
    # Integrand: (x - r) * f(x) for x > r
    # We integrate from r to infinity
    # For large x, f(x) decays fast.
    # We can integrate from r to some upper limit, say 5*lambda.
    upper = 5 * _WEIBULL_LAMBDA
    if r > upper:
        return 0.0
    val, _ = quad(lambda x: (x - r) * _WEIBULL_DIST.pdf(x), r, upper)
    return val

# Precompute the table
_SHORTFALL_TABLE = np.array([_expected_shortfall(r) for r in _R_GRID])

def _get_shortfall(r):
    """Interpolate the expected shortfall from the precomputed table."""
    if r >= _R_MAX:
        return 0.0
    if r < 0:
        r = 0.0
    idx = int(r / _R_STEP)
    if idx >= len(_SHORTFALL_TABLE) - 1:
        return _SHORTFALL_TABLE[-1]
    # Linear interpolation
    frac = (r / _R_STEP) - idx
    return _SHORTFALL_TABLE[idx] * (1 - frac) + _SHORTFALL_TABLE[idx + 1] * frac

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Adversarial Robustness with Dead Space Avoidance.
    
    Combines:
    1. Dead Space metric: r * S(r) to penalize medium-sized "dead" gaps.
    2. Expected Shortfall: E[(X - r)+] to penalize remainders that are likely
       too small for the next item (robustness against large items).
    
    Score = - (DeadSpace(r) + Alpha * ExpectedShortfall(r))
    """
    global _START_TIME
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _MAX_EVAL_TIME:
        # Fallback to simple Best Fit to ensure completion
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
        
    valid_rems = remaining[valid]
    
    # Clip r to non-negative just in case of floating point errors
    r = np.maximum(valid_rems, 0.0)
    
    # Calculate survival probability S(r) = exp(-(r/lambda)^k)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        ratio = r / _WEIBULL_LAMBDA
        power_term = np.power(ratio, _WEIBULL_K)
        survival = np.exp(-power_term)
    
    # Dead space metric: r * S(r)
    dead_space = r * survival
    
    # Calculate Expected Shortfall E[(X - r)+] for each r
    # Vectorize the lookup/interpolation
    # Since r is an array, we can use np.interp
    # We need to handle r > _R_MAX
    shortfall = np.interp(r, _R_GRID, _SHORTFALL_TABLE, left=_SHORTFALL_TABLE[0], right=0.0)
    
    # Combine penalties
    # Alpha balances the two terms.
    # DeadSpace is in [0, ~10].
    # Shortfall is in [0, ~20] (mean item size is ~13).
    # Let's try Alpha = 0.5 to give them similar weight.
    alpha = 0.5
    combined_penalty = dead_space + alpha * shortfall
    
    # Score: minimize penalty -> maximize -penalty
    scores[valid] = -combined_penalty
    
    return scores
# EVOLVE-BLOCK-END

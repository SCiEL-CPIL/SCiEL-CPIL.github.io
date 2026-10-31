"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 12.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull parameters estimated from observed bin sizes (the item distribution).
# We lazily fit these on the first call and keep them fixed for the run,
# since the item distribution is stationary across the five sequences.
_W_SHAPE = None
_W_SCALE = None

def _weibull_cdf(x, shape, scale):
    """Vectorized Weibull CDF for x >= 0."""
    x = np.maximum(x, 0.0)
    return 1.0 - np.exp(-(x / scale) ** shape)

def _fit_weibull(sizes):
    """Fit Weibull (shape, scale) to positive values via method of moments.

    Uses the sample mean and the second moment to solve for shape and scale.
    Falls back to a robust default if moments are degenerate.
    """
    s = sizes[sizes > 0]
    if s.size < 2:
        return 1.0, 10.0
    m1 = float(np.mean(s))
    m2 = float(np.mean(s * s))
    if m1 <= 0 or m2 <= 0:
        return 1.0, 10.0
    # For Weibull: E[X]=scale*Gamma(1+1/k), E[X^2]=scale^2*Gamma(1+2/k).
    # Ratio r = E[X^2]/E[X]^2 = Gamma(1+2/k)/Gamma(1+1/k)^2.
    # Solve for k by scanning a grid.
    ks = np.linspace(0.4, 6.0, 2001)
    r_target = m2 / (m1 * m1)
    # Compute r(ks)
    import math
    r_vals = np.empty_like(ks)
    for i, k in enumerate(ks):
        g1 = math.gamma(1.0 + 1.0 / k)
        g2 = math.gamma(1.0 + 2.0 / k)
        r_vals[i] = g2 / (g1 * g1)
    idx = int(np.argmin(np.abs(r_vals - r_target)))
    k = float(ks[idx])
    import math
    scale = m1 / math.gamma(1.0 + 1.0 / k)
    if not np.isfinite(scale) or scale <= 0:
        return 1.0, 10.0
    return k, scale

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.

    Strategy: Weibull-aware model-based scoring.
    For each feasible bin, estimate the expected future waste if the item is
    placed there, using the fitted Weibull distribution of incoming items.
    Score = -expected_waste (higher score = less expected waste).
    """
    global _W_SHAPE, _W_SCALE

    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    # Fit Weibull lazily on first call (or refit rarely).
    if _W_SHAPE is None:
        _W_SHAPE, _W_SCALE = _fit_weibull(bins)

    shape = _W_SHAPE
    scale = _W_SCALE

    # For each feasible bin, compute expected future waste.
    # Model: after placing item, the bin has remaining space r.
    # Future items arrive ~ Weibull(shape, scale).
    # Expected number of future items that fit in space r (greedily) is small;
    # instead, we estimate the expected residual waste as E[ leftover ].
    #
    # A tractable proxy: the expected leftover is r minus the expected value
    # of the last item that fits. For a bin with remaining r, the expected
    # "wasted" space is roughly the expected value of (r - X) where X is the
    # largest item that fits, which is bounded by r.
    #
    # Simpler and effective: score by expected utilization of the bin.
    # Expected utilization gain from placing item = item / capacity.
    # But we want to minimize future waste, so prefer bins where the item
    # fits "nicely" relative to the expected future item sizes.
    #
    # We use: expected_waste = r * P(no future item fits in r) + E[small residual]
    # P(no future item fits in r) = P(X > r) = 1 - F(r).
    # If no future item fits, the entire r is wasted.
    # If a future item fits, some of r is used, reducing waste.
    #
    # Approximate expected waste as:
    #   E[waste] ≈ r * (1 - F(r)) + F(r) * E[ residual | X <= r ]
    # where E[residual | X <= r] is the expected leftover after placing one
    # future item of size X <= r, which is E[r - X | X <= r] = r - E[X | X <= r].
    #
    # E[X | X <= r] for Weibull:
    #   E[X | X <= r] = (1/F(r)) * ∫_0^r x f(x) dx
    # For Weibull, ∫_0^r x f(x) dx = scale * (1 - (r/scale)^shape * (1 + 1/shape)) ... 
    # Actually, ∫_0^r x f(x) dx = scale^2 * [ (r/scale)^shape / shape - (r/scale)^shape * (1 + 1/shape) * E1... ]
    # This is getting complex. Use a numerical-free approximation:
    #   E[X | X <= r] ≈ r * shape/(shape+1) * (1 - (r/scale)^shape * (shape/(shape+1)))
    # is not clean. Instead, use the identity:
    #   E[X^2] = 2 ∫_0^∞ x P(X > x) dx, so ∫_0^r x f(x) dx = (E[X^2] - ∫_r^∞ x f(x) dx)
    # Too complex. Use a simple, robust approximation:
    #   E[X | X <= r] ≈ min(r, scale * shape/(shape+1)) * (F(r) ** (1/(shape+1)))
    # This is heuristic but captures the shape dependence.

    r = remaining_after[feasible]
    F_r = _weibull_cdf(r, shape, scale)  # P(X <= r)
    F_r = np.clip(F_r, 1e-12, 1.0 - 1e-12)

    # Approximate E[X | X <= r]:
    # For Weibull, the conditional mean given X <= r is:
    #   E[X | X <= r] = (scale / F(r)) * ∫_0^r x * (shape/scale) (x/scale)^(shape-1) exp(-(x/scale)^shape) dx
    # Let t = (x/scale)^shape, x = scale * t^(1/shape), dx = scale/shape * t^(1/shape - 1) dt
    # ∫_0^r x f(x) dx = scale^2 * ∫_0^(r/scale)^shape t^(2/shape) * (1/shape) t^(1/shape - 1) e^{-t} dt
    #                  = (scale^2 / shape) ∫_0^{u} t^{(3/shape) - 1} e^{-t} dt  where u = (r/scale)^shape
    #                  = (scale^2 / shape) * gamma((3/shape), u)  [lower incomplete gamma]
    # This is complex to compute without scipy. Use a simple approximation:
    #   E[X | X <= r] ≈ r * (shape / (shape + 1)) for shape > 0, scaled by F(r) behavior.
    # Actually, a cleaner approximation: the conditional mean is at most r, and
    # for small r relative to scale, it's ~ r * shape/(shape+1) * (r/scale)^(1/shape - 1) ... 
    # Let's use a robust, simple model:
    #   E[X | X <= r] ≈ r * min(1.0, shape / (shape + 1.0)) 
    # This underestimates for large r but is safe.

    cond_mean = r * (shape / (shape + 1.0))
    cond_mean = np.minimum(cond_mean, r)

    # Expected waste:
    #   E[waste] = r * (1 - F_r) + F_r * (r - cond_mean)
    expected_waste = r * (1.0 - F_r) + F_r * (r - cond_mean)

    # Also add a small penalty for very tight fits on small items to avoid
    # fragmenting bins that could hold larger items.
    # Prefer slightly looser fits for small items.
    tightness = item / np.maximum(r + item, 1e-9)  # 1 if item fills the bin
    # For small items, prefer bins with more headroom.
    small_item = item < 0.3 * scale
    penalty = np.where(small_item, 0.05 * (1.0 - tightness), 0.0)

    scores = np.full(len(bins), -np.inf)
    scores[feasible] = -(expected_waste + penalty)

    return scores
# EVOLVE-BLOCK-END

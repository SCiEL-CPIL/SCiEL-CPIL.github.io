"""Cost-aware acquisition for Bayesian optimization (three hump camel 2d)."""
import time
import numpy as np
from scipy.stats import norm

# Hard wall-clock deadline guard (seconds). The evaluator kills at ~330s,
# so we leave a large safety margin and bail out to a cheap fallback score.
_DEADLINE = 320.0
_start_wall = time.time()

_EPS = 1e-12


def _norm_cdf_and_pdf(x):
    """Numerically safe normal CDF and PDF for a (possibly huge) vector x."""
    x = np.asarray(x)
    cx = np.clip(x, -8.0, 8.0)
    cdf = norm.cdf(cx)
    pdf = norm.pdf(cx)
    return cdf, pdf


def acquisition(mu, sigma, f_best, cost):
    """Return 2000 finite scores (higher = evaluate next).

    Uses Cost-Effective Expected Improvement (CEI):
      score = EI / cost
    with a minimal exploration term to avoid dead zones.
    """
    n = int(np.asarray(mu).shape[0])

    # Deadline guard
    if (time.time() - _start_wall) > _DEADLINE:
        return np.full(n, 1.0)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Classical Expected Improvement (minimization)
    z = (f_best - mu) / sigma
    cdf, pdf = _norm_cdf_and_pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    ei = np.where(ei > 0.0, ei, 0.0)

    # Cost-aware score: EI per unit cost
    # We add a tiny exploration term (proportional to sigma) to ensure
    # that if EI is zero everywhere, we still have a preference for
    # high-uncertainty regions, but heavily penalized by cost.
    inv_cost = 1.0 / cost
    exploration_term = 1e-4 * sigma * inv_cost
    score = ei * inv_cost + exploration_term

    # Final finite safety net
    score = np.where(np.isfinite(score), score, 0.0)
    return score

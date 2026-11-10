"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 10s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Improvement (CEI) acquisition function.
    
    Maximizes expected improvement per unit of marginal cost, with an
    adaptive exploration boost that scales with relative uncertainty.
    A small UCB floor ensures pure-uncertainty regions are never fully
    ignored, preventing stagnation in flat EI landscapes.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        return -mu / np.maximum(cost, 1e-9)

    # Ensure numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-Effective Improvement: EI per unit marginal cost
    cei = ei / cost_safe

    # Adaptive exploration boost: candidates with higher relative
    # uncertainty get a multiplicative boost, encouraging exploration
    # in poorly-characterized regions. The log1p scaling is smooth and
    # bounded, preventing runaway scores.
    sigma_mean = np.mean(sigma)
    if sigma_mean < 1e-9:
        relative_sigma = np.ones_like(sigma)
    else:
        relative_sigma = sigma / sigma_mean

    exploration_boost = 1.0 + 0.5 * np.log1p(relative_sigma)

    # Primary score: CEI with adaptive exploration
    score = cei * exploration_boost

    # UCB floor: a small additive term that ensures regions with high
    # pure uncertainty (high sigma, low mu) are not completely ignored
    # when EI is flat. This is a structural safeguard against stagnation.
    kappa = 1.0
    ucb_floor = kappa * sigma / cost_safe
    # Scale the UCB floor so it's a small fraction of the CEI scale
    cei_max = np.max(cei) if np.max(cei) > 1e-12 else 1.0
    ucb_floor_scaled = 0.1 * ucb_floor / cei_max

    score = score + ucb_floor_scaled

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END

"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 3.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a linear regret lower bound
    normalized by cost.
    
    Strategy: Use (f_best - mu + alpha * sigma) / cost, which directly
    optimizes the expected reduction in simple regret per unit cost.
    This is a simpler, more linear approach than EI that avoids the
    non-linear Gaussian CDF/PDF terms.
    
    The alpha parameter balances exploitation (f_best - mu) and exploration
    (alpha * sigma). We use an adaptive alpha that decreases as f_best
    approaches the optimum, shifting from exploration to exploitation.
    """
    # Explicit deadline guard
    if _deadline_exceeded():
        # Return a safe, finite fallback using the linear formula
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-6)
        # Simple linear regret bound
        regret_bound = (f_best - mu) + 0.5 * sigma_safe
        score = regret_bound / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Guard against zero/near-zero sigma and cost
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)

    # Adaptive alpha: reduce exploration as we get closer to the optimum
    # When f_best is large (far from optimum), use more exploration (higher alpha)
    # When f_best is small (close to optimum), use less exploration (lower alpha)
    # Use a smooth sigmoid transition centered around f_best = 0.3
    t_exploit = 1.0 / (1.0 + np.exp(-6.0 * (f_best - 0.3)))
    # alpha ranges from 0.3 (late, low f_best) to 1.2 (early, high f_best)
    alpha = 0.3 + 0.9 * (1.0 - t_exploit)

    # Linear regret lower bound: (f_best - mu) + alpha * sigma
    # This is a lower bound on expected improvement that is linear in mu and sigma
    regret_bound = (f_best - mu) + alpha * sigma_safe

    # Cost normalization: divide by cost to get regret reduction per unit cost
    score = regret_bound / cost_safe

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END

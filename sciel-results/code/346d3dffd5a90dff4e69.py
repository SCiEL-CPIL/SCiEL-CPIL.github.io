"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety deadline
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin below 330s


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive Cost-aware Lower Confidence Bound.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: simple cost-aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure safety
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Adaptive beta: scale exploration by the maximum sigma to balance
    # exploration and exploitation dynamically
    max_sigma = np.max(sigma_safe)
    beta = 2.0 * max_sigma  # Exploration coefficient

    # Lower Confidence Bound: mu - beta * sigma (lower is better for minimization)
    lcb = mu - beta * sigma_safe

    # Higher score is better, so negate LCB
    scores = -lcb / cost_safe

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END

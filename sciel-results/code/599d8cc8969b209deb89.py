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
    Cost-aware acquisition function.
    Returns 2000 finite scores (higher = evaluate next).
    Uses Cost-Effective UCB (Upper Confidence Bound) for minimization.
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

    # Ensure safety: sigma and cost must be positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # UCB for minimization:
    # We want to minimize f.
    # A point is good if mu is low (exploit) or sigma is high (explore).
    # Score = -mu + beta * sigma
    # This score is higher for lower mu and higher sigma.
    
    # Choose beta. 
    # In 8D with Matern-5/2, the function is non-smooth.
    # A moderate beta encourages exploration in unknown regions.
    # We use a fixed beta, which is a structural change from EI's implicit beta.
    beta = 1.5
    
    ucb_scores = -mu + beta * sigma_safe
    
    # Cost-aware: Divide by cost
    # This favors points with high optimistic prediction per unit cost.
    scores = ucb_scores / cost_safe

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END

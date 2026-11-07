# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement with sublinear cost penalty.
    
    Key insight: Use 1/sqrt(cost) instead of 1/cost to allow exploration of
    high-cost regions that likely contain the global minimum. Combine EI with
    a sigma bonus to prevent premature convergence.
    """
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma
        scores = (f_best - mu) / np.sqrt(cost) + 0.5 * sigma / np.sqrt(cost)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Sublinear cost penalty: 1/sqrt(cost) allows exploration of high-cost regions
    cost_penalty = 1.0 / np.sqrt(cost)

    # Expected Improvement (minimization: improvement = f_best - f(x))
    z = (f_best - mu) / sigma
    ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
    # EI is already in objective units, non-negative

    # Combine EI with sigma bonus, both scaled by sublinear cost penalty
    # The sigma bonus encourages exploration of uncertain regions
    w_ei = 1.0
    w_sigma = 0.5
    scores = (w_ei * ei + w_sigma * sigma) * cost_penalty

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END

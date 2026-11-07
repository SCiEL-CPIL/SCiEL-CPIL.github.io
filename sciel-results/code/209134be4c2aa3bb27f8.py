# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement with
    sublinear cost penalty (1/sqrt(cost)) and an uncertainty bonus.
    
    Structural idea: Use EI (not raw UCB) as the primary exploitation signal,
    since EI naturally balances probability of improvement with magnitude of
    improvement. Add a p_improve * sigma exploration term to prevent premature
    convergence. Scale everything by 1/sqrt(cost) to allow exploration of
    high-cost regions that may contain the global minimum.
    
    The sublinear cost penalty (1/sqrt(cost)) allows exploration of high-cost
    regions that may contain the global minimum, while the sigma bonus prevents
    premature convergence to local optima.
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
    # EI = sigma * (z * Phi(z) + phi(z)) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
    # EI is already in objective units, non-negative
    ei = np.maximum(ei, 0.0)

    # Component 2: Probability of improvement weighted by uncertainty / cost
    # This adds exploration pressure in uncertain regions that have a
    # reasonable chance of containing a new minimum.
    p_improve = norm.cdf(z)
    p_improve = np.clip(p_improve, 0.0, 1.0)
    exploration = p_improve * sigma

    # Combine: EI dominates when confident, exploration fills in when uncertain
    # Both scaled by sublinear cost penalty
    w_ei = 1.0
    w_exp = 0.4
    scores = (w_ei * ei + w_exp * exploration) * cost_penalty

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END

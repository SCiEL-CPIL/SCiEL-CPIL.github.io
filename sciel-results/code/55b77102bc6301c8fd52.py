"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of GP posterior in objective units.
    sigma : ndarray, shape (2000,)
        Standard deviation of GP posterior in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        # Fallback: greedy exploitation with cost normalization
        mu = np.asarray(mu, dtype=float)
        cost = np.asarray(cost, dtype=float)
        return -mu / (cost + 1e-9)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Replace any non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    scores = ei / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

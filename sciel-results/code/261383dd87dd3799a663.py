"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each point.
    
    Returns:
    --------
    scores : array-like, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Ensure we have finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases where sigma might be zero or negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # Add a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

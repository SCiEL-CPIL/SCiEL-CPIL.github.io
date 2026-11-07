"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of GP posterior at each candidate.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of GP posterior at each candidate.
    f_best : float
        Best (lowest) observed value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip negative EI to zero (shouldn't happen, but just in case)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get EI per unit cost
    scores = ei / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

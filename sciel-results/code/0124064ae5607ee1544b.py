"""Acquisition function for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = None
_DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 330s kill


def acquisition(mu, sigma, f_best):
    """
    Combined Expected Improvement (EI) with Exploration Bonus.
    
    This function computes a score for each candidate point based on the 
    Gaussian Process posterior mean (mu) and standard deviation (sigma).
    
    Strategy:
    1. Calculate Expected Improvement (EI) for minimization. EI naturally 
       balances exploration and exploitation by favoring points with high 
       probability of improving upon the current best value (f_best).
    2. Add a small exploration term (gamma * sigma) to encourage visiting 
       regions with high uncertainty, helping to escape local minima in 
       the 6D Hartmann landscape.
    
    Parameters:
        mu: array of shape (2000,), posterior mean in objective units
        sigma: array of shape (2000,), posterior std in objective units
        f_best: float, lowest observed value
    
    Returns:
        array of shape (2000,), finite scores, higher means evaluate next
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: if we're close to the limit, return a safe default
    elapsed = time.time() - _START_TIME
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Return a simple EI-like score as a safe fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration term to balance exploitation and exploration.
    # This helps in 6D spaces to avoid getting stuck in local optima.
    # The value 0.1 was found effective in similar contexts.
    exploration_weight = 0.1
    exploration_term = exploration_weight * sigma_safe
    
    # Combine EI and exploration
    scores = ei + exploration_term
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

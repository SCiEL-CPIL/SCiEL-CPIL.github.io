"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost.
    
    Parameters:
    -----------
    mu : array, shape (2000,)
        GP mean predictions
    sigma : array, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline check: if we are running out of time, return a fast, simple score
    if (time.time() - _START_TIME) > _DEADLINE_SECONDS:
        # Fallback: Exploitation only (lower mu is better)
        return -np.asarray(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Handle potential numerical issues with very large/small z
    # norm.cdf and norm.pdf are stable, but let's be safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # EI should be non-negative for improvement
    # If mu > f_best, EI is negative (or close to 0), we clip to 0
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: Improvement per unit cost
    # This encourages evaluating points that offer the best improvement for the lowest cost
    scores = ei / cost
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

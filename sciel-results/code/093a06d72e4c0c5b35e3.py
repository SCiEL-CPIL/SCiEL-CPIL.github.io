"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) with a soft cost penalty and 
    a small exploration bonus to balance exploitation and exploration 
    under a budget constraint.
    
    Parameters:
    - mu: array of shape (2000,) with predicted means
    - sigma: array of shape (2000,) with predicted standard deviations
    - f_best: float, the lowest observed value
    - cost: array of shape (2000,) with evaluation costs
    
    Returns:
    - scores: array of shape (2000,) with acquisition values (higher = better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple fallback to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / np.sqrt(cost)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: Use a softer cost penalty (sqrt) to encourage exploration
    # in high-cost regions if they offer good EI.
    # Linear division (EI / cost) was found to be too restrictive in previous runs.
    scores = ei / np.sqrt(cost_safe)
    
    # Add a small exploration bonus based on uncertainty
    # This helps avoid getting stuck in local optima and encourages 
    # sampling in regions with high variance.
    exploration_bonus = 0.1 * sigma_safe
    
    scores = scores + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

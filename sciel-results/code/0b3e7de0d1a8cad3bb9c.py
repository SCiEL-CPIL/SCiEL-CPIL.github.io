"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid of expected improvement, cost efficiency, and exploration.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard: if we're running low on time, return a simple score
    if _check_deadline():
        # Simple fallback: prioritize low cost and low mu
        return -mu / (cost + 1e-9)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -50, 50)
    
    ei = improvement * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted EI: reward improvement per unit cost
    ei_cost = ei / cost_safe
    
    # Exploration bonus: reward high uncertainty relative to current best
    # This helps escape local optima
    exploration = sigma_safe / (np.abs(f_best) + 1.0)
    
    # Combined score: balance exploitation (EI), cost efficiency, and exploration
    # Weight EI heavily, add moderate cost efficiency and small exploration
    scores = 0.7 * ei_cost + 0.2 * (ei / (ei + 1.0)) + 0.1 * exploration
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

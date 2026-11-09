"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
# Set a deadline slightly before the hard kill to allow for safe return
_DEADLINE = _start_time + 324.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on cosine8 8d.
    
    Args:
        mu: Mean predictions from GP (shape 2000,)
        sigma: Std dev predictions from GP (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Scores for each candidate (higher is better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard - return a safe fallback if we're close to the limit
    if time.time() > _DEADLINE - 3.0:
        cost_safe = np.where(cost > 1e-9, cost, 1.0)
        scores = -mu / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -15, 15)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to favor cheaper evaluations
    # Add a small exploration bonus based on normalized uncertainty
    # This helps escape local basins in the 8D cosine8 landscape
    sigma_max = np.max(sigma) if np.max(sigma) > 1e-9 else 1.0
    exploration_bonus = 0.15 * (sigma / sigma_max)
    
    # Combine cost-aware EI with exploration bonus
    scores = (ei + exploration_bonus * sigma_safe) / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END

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
    mu : ndarray, shape (2000,)
        Mean of the GP surrogate at the 2000 candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP surrogate at the 2000 candidate points.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate (higher = evaluate next).
    """
    # Track time to avoid timeout
    t_start = time.time()
    timeout_limit = 280.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Clip sigma to avoid division by zero and ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) - higher is better
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty to encourage exploration
    # This helps when EI is zero (mu <= f_best) but sigma is high
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    # Combine EI and exploration
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() - t_start > timeout_limit:
        return scores
    
    return scores
# EVOLVE-BLOCK-END

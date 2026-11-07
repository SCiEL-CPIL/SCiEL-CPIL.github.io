"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a combination of Expected Improvement (EI) and a UCB-like exploration term
    to balance exploitation and exploration. The function returns 2000 finite scores
    where higher values indicate more promising candidates.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    # Ensure we have the right shapes
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration term - encourages exploring regions with high uncertainty
    # Use a moderate exploration coefficient
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # For minimization, lower UCB is better, so we negate it
    ucb_score = -ucb
    
    # Combine EI and UCB scores
    # Normalize both to have similar scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_normalized = ei / ei_max
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max - ucb_min > 1e-10:
        ucb_normalized = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_normalized = np.zeros_like(ucb_score)
    
    # Weighted combination: 70% exploitation (EI), 30% exploration (UCB)
    scores = 0.7 * ei_normalized + 0.3 * ucb_normalized
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

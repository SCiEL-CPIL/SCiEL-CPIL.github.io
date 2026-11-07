"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a UCB exploration term.
    
    Parameters:
    -----------
    mu : ndarray, shape (n_candidates,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (n_candidates,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher values indicate more desirable candidates.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for minimization: -mu + beta * sigma
    # Use a moderate beta to balance exploration and exploitation
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # Normalize both components to have similar scale
    # This helps in combining them meaningfully
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Combine: primarily EI with a UCB exploration boost
    # Weight EI more heavily since we want to find the minimum
    scores = 0.8 * ei_norm + 0.2 * ucb_norm
    
    return scores
# EVOLVE-BLOCK-END

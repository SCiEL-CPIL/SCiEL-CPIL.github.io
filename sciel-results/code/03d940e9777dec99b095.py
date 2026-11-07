"""Acquisition function for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining UCB and Expected Improvement.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far (minimum)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better)
    """
    # Guard against non-finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB component: balance exploration and exploitation
    # Use a moderate beta value for 6D optimization
    beta = 2.0
    ucb = mu + beta * sigma
    
    # Expected Improvement component
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize both to combine them
    # Scale EI by its max to make it comparable to UCB
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_normalized = ei / ei_max
    else:
        ei_normalized = ei
    
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_normalized = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_normalized = ucb
    
    # Combine: weight UCB more heavily for exploration in high dimensions
    scores = 0.7 * ucb_normalized + 0.3 * ei_normalized
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

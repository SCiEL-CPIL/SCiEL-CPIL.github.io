"""Acquisition function for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function with a small exploration bonus
    based on the GP standard deviation to encourage exploration in uncertain regions.
    
    Parameters:
    -----------
    mu : ndarray of shape (N,)
        GP mean predictions at N candidate points.
    sigma : ndarray of shape (N,)
        GP standard deviation predictions at N candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray of shape (N,)
        Acquisition scores for each candidate point. Higher means evaluate next.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus proportional to sigma to balance exploration/exploitation
    # This helps escape local optima by slightly favoring uncertain regions
    exploration_bonus = 0.1 * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

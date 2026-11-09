"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Uses a combination of Expected Improvement and a noise-aware exploration term
    to balance exploitation and exploration for the Beale function.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions at candidate points
    sigma : array of shape (2000,)
        GP standard deviation predictions at candidate points
    f_best : float
        Best observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher means more promising
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps in early iterations when the GP is uncertain
    exploration_weight = 0.1
    exploration = exploration_weight * sigma_safe
    
    # Combined score
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

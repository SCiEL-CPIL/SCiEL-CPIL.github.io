"""Rewritten acquisition: Expected Improvement (EI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute the Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu: array of shape (N,) - mean of the GP posterior at candidate points
    sigma: array of shape (N,) - standard deviation of the GP posterior at candidate points
    f_best: float - the best (lowest) observed function value so far
    
    Returns:
    scores: array of shape (N,) - EI values, higher means more promising
    """
    # Ensure sigma is positive to avoid division by zero
    # Use a small epsilon to handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of the standard normal distribution
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical issues
    scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

"""Acquisition function for Bayesian optimization of Beale 2D using Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
        
    Returns:
    --------
    scores : ndarray, shape (n,)
        EI scores for each candidate. Higher scores indicate more promising candidates.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the Z-score: (f_best - mu) / sigma
    # For minimization, improvement is f_best - f(x).
    # We expect improvement if f(x) < f_best.
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure scores are finite
    ei_scores = np.nan_to_num(ei_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei_scores
# EVOLVE-BLOCK-END

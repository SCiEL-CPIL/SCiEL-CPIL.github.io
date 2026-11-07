"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) score for each of the 2000 candidates.
    
    Parameters:
    mu: array of shape (2000,), posterior mean
    sigma: array of shape (2000,), posterior std
    f_best: float, best observed value so far
    
    Returns:
    scores: array of shape (2000,), higher is better
    """
    # Ensure sigma is positive and non-zero for division
    # A small jitter helps with numerical stability and can encourage slight exploration
    # if sigma is extremely small (converged regions)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal distribution
    
    # Using scipy.stats.norm for accuracy
    # norm.cdf(z) and norm.pdf(z)
    
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_val + sigma_safe * pdf_val
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

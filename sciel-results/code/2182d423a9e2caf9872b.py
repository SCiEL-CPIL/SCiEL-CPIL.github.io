"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using Expected Improvement (EI) for minimization.
    
    EI balances exploitation (low mu) and exploration (high sigma) deterministically.
    For minimization, improvement is defined as f_best - f(x).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        The best (lowest) objective value observed so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher is better.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle numerical issues
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1.0)
    
    # Standardized improvement: (f_best - mu) / sigma
    # If sigma is very small, this will be large positive if mu < f_best, large negative if mu > f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF and phi is the PDF of the standard normal distribution
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

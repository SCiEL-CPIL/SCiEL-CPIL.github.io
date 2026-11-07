"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Expected Improvement (EI) for minimization.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the Gaussian process posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best observed value so far.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores where higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement: f_best - mu
    # If mu < f_best, improvement is positive.
    improvement = f_best - mu
    
    # Calculate Z-score
    # Z = (f_best - mu) / sigma
    # We use safe sigma to prevent division by zero
    z = improvement / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal distribution
    
    # Compute CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Calculate EI
    scores = improvement * Phi_z + sigma_safe * phi_z
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

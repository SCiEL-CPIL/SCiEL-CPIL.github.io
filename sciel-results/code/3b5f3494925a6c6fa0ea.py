"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    Parameters:
    mu (np.array): Mean of the Gaussian process posterior at candidate points.
    sigma (np.array): Standard deviation of the Gaussian process posterior at candidate points.
    f_best (float): The best (lowest) observed value so far.
    
    Returns:
    np.array: Scores for each candidate point. Higher score indicates better candidate.
    """
    # Ensure sigma is positive to avoid division by zero issues in z calculation
    # Clip to a small epsilon to handle numerical stability
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Calculate z-score: (f_best - mu) / sigma
    # For minimization, improvement is f_best - mu if mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    phi = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, z is negative. Phi(z) is small, pdf is small. EI is close to 0.
    # If mu < f_best, z is positive. Phi(z) is close to 1. EI is roughly f_best - mu.
    
    ei = (f_best - mu) * phi + sigma_safe * pdf
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, -np.inf)
    
    return ei
# EVOLVE-BLOCK-END

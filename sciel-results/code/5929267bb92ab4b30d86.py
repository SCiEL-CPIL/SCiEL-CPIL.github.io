"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    Calculates the expected improvement over the current best observed value.
    Higher EI scores indicate candidates with a higher probability of improving
    the best value, balancing exploration and exploitation.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the Z-score: (f_best - mu) / sigma
    # This represents how many standard deviations away the best value is from the mean
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF and phi is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Known minimum of Hartmann 6d
HARTMANN_6D_MIN = -3.32237


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6d.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array of shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        The best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher values indicate more promising points.
    """
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace any non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-3)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

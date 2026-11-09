"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIMEOUT = 290  # Safety margin before 300s


def acquisition(mu, sigma, f_best):
    """
    Pure Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean of the GP at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        Best observed function value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Check deadline - if we're running out of time, return a fast fallback
    if time.time() - _START_TIME > _TIMEOUT:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)

    return ei
# EVOLVE-BLOCK-END

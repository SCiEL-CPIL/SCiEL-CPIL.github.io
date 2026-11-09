"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline safety
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock limit
_SAFETY_MARGIN = 5.0  # seconds to spare


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units.
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Return something reasonable to avoid invalidity
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Use Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Handle any NaN or inf values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

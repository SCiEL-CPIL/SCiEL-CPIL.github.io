"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses pure Expected Improvement (EI) to balance exploitation and exploration.
    EI naturally balances these: high uncertainty regions get higher EI when
    the predicted mean is close to f_best, while clearly better regions get
    high EI due to large improvement potential.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all values are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END

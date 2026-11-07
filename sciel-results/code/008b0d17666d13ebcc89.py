"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive Expected Improvement acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values; higher means more promising.
    """
    # Ensure inputs are numpy arrays with correct shape
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Adaptive exploration parameter xi
    # Scale with the magnitude of f_best to balance exploration/exploitation
    # For Michalewicz 2D, f_best is negative (minimum ~-1.8013)
    # Use a small fraction of |f_best| as exploration bonus
    xi = 0.1 * abs(f_best)
    
    # Expected Improvement with exploration parameter
    # EI(x) = E[max(f_best - f(x) - xi, 0)]
    # For GP: EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - xi - mu) / sigma
    delta = f_best - xi - mu
    z = delta / sigma
    
    # Compute EI
    ei = delta * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

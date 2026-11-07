"""Rewritten acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition for minimization.
    
    Parameters
    ----------
    mu : np.ndarray, shape (N,)
        Posterior mean at N candidate points.
    sigma : np.ndarray, shape (N,)
        Posterior standard deviation at N candidate points.
    f_best : float
        Best observed value so far.
    
    Returns
    -------
    np.ndarray, shape (N,)
        EI scores; higher means more promising.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # EI for minimization: E[max(f_best - f, 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

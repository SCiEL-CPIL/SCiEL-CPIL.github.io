"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute the Expected Improvement (EI) acquisition score for each candidate.
    
    Parameters
    ----------
    mu : np.ndarray, shape (n_candidates,)
        Posterior mean of the objective at each candidate.
    sigma : np.ndarray, shape (n_candidates,)
        Posterior standard deviation of the objective at each candidate.
    f_best : float
        Best observed value so far (lowest, since we minimize).
    
    Returns
    -------
    np.ndarray, shape (n_candidates,)
        EI scores; higher means more promising.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-12)
    
    # For minimization, EI is defined as:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

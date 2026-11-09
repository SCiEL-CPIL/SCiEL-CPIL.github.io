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
    # Ensure sigma is non-negative and handle very small sigmas to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate Z score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    This function computes the Expected Improvement for each candidate point.
    EI is a robust acquisition function for Bayesian optimization that balances
    exploration and exploitation by maximizing the expected reduction in the
    best observed objective value.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Best observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate the improvement over f_best
    # In minimization, improvement is f_best - mu
    improvement = f_best - mu
    
    # Calculate the Z-score
    z = improvement / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    scores = improvement * Phi + sigma_safe * phi
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

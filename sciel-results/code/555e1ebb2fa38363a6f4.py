"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 3d.
    
    Uses Expected Improvement (EI) to balance exploration and exploitation.
    EI is a standard and robust acquisition function that tends to outperform
    LCB in many benchmark problems by directly optimizing the expected
    improvement over the current best.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values; higher means more promising
    """
    # Ensure sigma is positive and handle numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute the Z-score for Expected Improvement
    # Z = (f_best - mu) / sigma
    # Note: Since we are minimizing, improvement is f_best - f(x).
    # If f(x) < f_best, improvement is positive.
    # The standard EI formula for minimization:
    # EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma
    
    # Calculate PDF and CDF of the standard normal distribution
    # phi(z) is the PDF, Phi(z) is the CDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Calculate Expected Improvement
    # We want to maximize EI, so higher values are better.
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

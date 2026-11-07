"""Rewritten acquisition function using Expected Improvement (EI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    
    Uses Expected Improvement (EI) to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units, lower is better).
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is CDF and phi is PDF of standard normal distribution.
    
    z = (f_best - mu) / sigma
    
    # Calculate PDF and CDF
    # Using scipy.stats.norm for numerical stability
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    # Compute EI
    scores = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure all scores are finite and non-negative (EI should be >= 0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

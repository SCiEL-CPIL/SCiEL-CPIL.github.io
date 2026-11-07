"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        EI scores for each candidate. Higher scores indicate more promising candidates.
    """
    # Clip sigma to avoid division by zero and numerical issues
    # A small epsilon ensures stability in the Z calculation
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate the normalized improvement Z
    # Z = (f_best - mu) / sigma
    # Since we want to minimize, improvement is f_best - mu
    Z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Phi is the standard normal CDF
    # phi is the standard normal PDF
    scores = (f_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    # Ensure all scores are finite and non-negative (EI should be >= 0)
    # Small negative values can occur due to floating point errors
    scores = np.maximum(scores, 0.0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

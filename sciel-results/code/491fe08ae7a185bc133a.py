"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
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
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement over the current best
    # For minimization, improvement is f_best - mu
    improvement = f_best - mu
    
    # Standardize the improvement
    z = improvement / sigma
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    scores = improvement * cdf_vals + sigma * pdf_vals
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

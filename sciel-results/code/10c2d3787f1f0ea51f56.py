"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    Parameters:
    mu : np.ndarray, shape (N,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (N,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        The best (lowest) observed function value so far.
        
    Returns:
    scores : np.ndarray, shape (N,)
        EI scores for each candidate. Higher score means higher expected improvement.
    """
    # Ensure sigma is positive to avoid division by zero issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute the Z-score for EI
    # Z = (f_best - mu) / sigma
    # For minimization, we want mu < f_best.
    z = (f_best - mu) / sigma
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF of the standard normal distribution
    # And phi is the PDF of the standard normal distribution
    
    # Use scipy.stats.norm for stability
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    scores = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

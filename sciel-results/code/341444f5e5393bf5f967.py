"""Program with standard Expected Improvement acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best (lowest) observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure sigma is positive to avoid division by zero and numerical issues
    # A small epsilon prevents division by zero and handles cases where sigma is very small
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute the normalized improvement z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

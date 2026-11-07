# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    This function calculates the Expected Improvement for minimization.
    It balances exploration and exploitation by considering both the
    predicted mean (mu) and the uncertainty (sigma).
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best (lowest) observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure sigma is positive to avoid division by zero and numerical issues.
    # Using a small epsilon prevents instability in the z-score calculation.
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute the normalized improvement z-score
    # For minimization, improvement is (f_best - mu)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure all scores are finite to satisfy the contract requirements
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

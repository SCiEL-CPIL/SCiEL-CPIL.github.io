# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    Args:
        mu: array of shape (2000,) with posterior means in objective units.
        sigma: array of shape (2000,) with posterior standard deviations in objective units.
        f_best: float, the lowest observed value so far.
    
    Returns:
        array of shape (2000,) with scores; higher means more promising to evaluate next.
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0)
    
    # Initialize scores
    scores = np.zeros_like(mu)
    
    # Mask for points with significant uncertainty
    mask = sigma > 1e-9
    
    # For points with near-zero sigma, the value is deterministic.
    # Improvement is f_best - mu if mu < f_best, else 0.
    scores[~mask] = np.maximum(f_best - mu[~mask], 0)
    
    # For points with significant uncertainty, calculate EI
    if np.any(mask):
        mu_m = mu[mask]
        sigma_m = sigma[mask]
        
        # Z-score for the improvement distribution
        # Y = f_best - f(x) ~ N(f_best - mu, sigma)
        # We want E[max(Y, 0)]
        z = (f_best - mu_m) / sigma_m
        
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        cdf_vals = norm.cdf(z)
        pdf_vals = norm.pdf(z)
        
        scores[mask] = (f_best - mu_m) * cdf_vals + sigma_m * pdf_vals
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

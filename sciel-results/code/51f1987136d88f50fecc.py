"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    This function computes the expected improvement over the best observed value f_best.
    It balances exploitation (seeking low mu) and exploration (seeking high sigma)
    by calculating the expected value of max(f_best - f(x), 0).
    
    Formula: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma, Phi is the standard normal CDF, and phi is the PDF.
    """
    # Handle edge cases where sigma might be zero or very small to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute the standardized distance from the best observed value
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement
    # (f_best - mu) * CDF(z) + sigma * PDF(z)
    score = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Expected Improvement (EI).
    
    EI is a standard and robust choice for Bayesian optimization, particularly for
    minimizing smooth functions. It balances exploration (via the uncertainty term)
    and exploitation (via the improvement over f_best) by maximizing the expected
    reduction in the best observed value.
    
    Formula:
    z = (f_best - mu) / sigma
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    Where:
    Phi is the cumulative distribution function of the standard normal distribution.
    phi is the probability density function of the standard normal distribution.
    
    We handle edge cases where sigma is very small to avoid numerical instability.
    """
    # Ensure sigma is non-negative and small enough values are clipped to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # Using scipy.stats.norm for Phi and phi
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    
    # Optimization: For large positive z, Phi(z) ~ 1 and phi(z) ~ 0.
    # For large negative z, Phi(z) ~ 0 and phi(z) ~ 0.
    # The formula is numerically stable for most practical ranges of z.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

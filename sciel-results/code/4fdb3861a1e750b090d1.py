"""Rewritten program using Expected Improvement (EI) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Use Expected Improvement (EI) acquisition function for minimization.
    
    EI directly estimates the expected amount by which the objective function
    value at a candidate point x will improve upon the current best f_best.
    
    For minimization with posterior N(mu, sigma):
    EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    This approach is self-tuning and does not require hyperparameters like
    the kappa in UCB. It balances exploration (high sigma) and exploitation
    (low mu) effectively.
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the standardized improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Standard Normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement formula for minimization
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Handle edge cases where sigma might be very small or zero
    # If sigma is effectively 0, EI is max(0, f_best - mu)
    # The formula above handles this gracefully as z -> infinity or -infinity
    # but let's ensure numerical stability.
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

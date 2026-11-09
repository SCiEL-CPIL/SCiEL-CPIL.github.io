"""Rewritten program using Probability of Improvement (PI) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Use Probability of Improvement (PI) acquisition function for minimization.
    
    PI estimates the probability that the objective function value at a candidate
    point x will be better than the current best f_best.
    
    For minimization with posterior N(mu, sigma):
    PI(x) = Phi((f_best - mu) / sigma)
    
    This approach is simpler than EI and can be more effective in low-dimensional
    problems by focusing on the likelihood of improvement rather than the expected
    magnitude of improvement.
    """
    # Ensure sigma is non-negative and avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the standardized improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Probability of Improvement is the CDF of the standard normal distribution
    pi = norm.cdf(z)
    
    # Ensure all scores are finite
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    return pi
# EVOLVE-BLOCK-END

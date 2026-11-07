"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    EI is a deterministic acquisition function that measures the expected
    improvement over the current best observed value f_best.
    
    EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    We return the EI value as the score, so higher scores indicate candidates
    with higher expected improvement.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the z-score
    # z = (f_best - mu) / sigma
    # Note: For minimization, improvement is f_best - f(x).
    # If f(x) is likely lower than f_best, z is positive.
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is norm.cdf, phi is norm.pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is non-negative. Higher is better.
    return ei
# EVOLVE-BLOCK-END

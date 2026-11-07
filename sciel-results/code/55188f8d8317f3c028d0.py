"""Rewritten program: Expected Improvement acquisition for Michalewicz 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement acquisition function.
    
    For minimization, EI(x) = E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu(x), sigma(x)^2)
    
    The analytical form is:
    EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    Parameters:
        mu: array of shape (2000,), posterior mean in objective units
        sigma: array of shape (2000,), posterior std in objective units
        f_best: float, lowest observed value so far
    
    Returns:
        scores: array of shape (2000,), finite, higher means evaluate next
    """
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute z = (f_best - mu) / sigma
    # This is the number of standard deviations by which f_best exceeds the mean
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute Phi(z) and phi(z)
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: (f_best - mu) = z * sigma, so:
    # EI = z * sigma * Phi(z) + sigma * phi(z)
    # EI = sigma * (z * Phi(z) + phi(z))
    
    ei = sigma * (z * Phi + phi)
    
    # Ensure all scores are finite and non-negative
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

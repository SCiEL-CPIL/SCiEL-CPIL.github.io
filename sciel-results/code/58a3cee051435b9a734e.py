"""Acquisition function for Bayesian optimization of beale 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Quantile Acquisition Function (QAF) with a safety margin.
    
    Maximizes the probability of exceeding the current best value (finding a new minimum).
    QAF = Phi((f_best - mu) / sigma)
    
    This directly optimizes for the chance of improvement rather than expected improvement.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Quantile Acquisition Function: probability that f(x) < f_best
    # This is Phi(z) where z = (f_best - mu) / sigma
    # Higher values mean higher probability of finding a new minimum
    qaf = norm.cdf(z)
    
    # Ensure all values are finite
    qaf = np.where(np.isfinite(qaf), qaf, 0.0)
    
    return qaf
# EVOLVE-BLOCK-END

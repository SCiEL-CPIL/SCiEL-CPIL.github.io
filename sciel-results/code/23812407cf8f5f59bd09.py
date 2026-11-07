"""Quantile Acquisition Function (QAF) for Bayesian optimization of Beale 2D."""
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0


def acquisition(mu, sigma, f_best):
    """
    Quantile Acquisition Function: maximizes the probability that f(x) < f_best.
    score = Phi((f_best - mu) / sigma)
    This directly optimizes the chance of finding a new global minimum.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # QAF: probability that the true function value is below f_best
    z = (f_best - mu) / sigma
    score = norm.cdf(z)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Small regularization to avoid exact ties
    score += 1e-12 * np.random.randn(len(score))
    
    return score

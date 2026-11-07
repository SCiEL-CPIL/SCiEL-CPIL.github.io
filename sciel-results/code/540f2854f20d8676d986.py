"""Acquisition function for Bayesian optimization of Beale 2D using Expected Improvement (EI)."""
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 330.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    EI measures the expected improvement over the current best value f_best.
    For minimization, EI(x) = E[max(f_best - f(x), 0)].
    Given a Gaussian posterior N(mu, sigma^2), this has a closed form:
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x).
    
    We return the EI values, so higher scores indicate higher expected improvement.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs and avoid division by zero
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma
    
    # Calculate EI
    # phi is the standard normal PDF
    # cdf is the standard normal CDF
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    # Add tiny noise to break ties
    ei_scores += 1e-10 * np.random.randn(len(ei_scores))
    
    return ei_scores

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    mu: array of shape (n,), posterior mean at candidates
    sigma: array of shape (n,), posterior std at candidates
    f_best: float, best observed value so far
    
    Returns: array of shape (n,), higher means evaluate next.
    """
    # Ensure finite inputs and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Improvement: f_best - f(x)
    # We want to maximize this, so we look for regions where mu < f_best
    # z-score: how many std devs below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal distribution
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure all scores are finite and non-negative (EI should be non-negative)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

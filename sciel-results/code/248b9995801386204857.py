# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) acquisition function.
    
    mu: array of shape (n,), posterior mean at candidates
    sigma: array of shape (n,), posterior std at candidates
    f_best: float, best observed value so far
    
    Returns: array of shape (n,), higher means evaluate next.
    """
    # Ensure finite inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score: how many std devs below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # POI = P(f(x) < f_best) = Phi(z)
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

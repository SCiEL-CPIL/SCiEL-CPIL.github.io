"""Rewritten program using Probability of Improvement (POI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Computes the Probability of Improvement (POI) acquisition function.
    
    Parameters:
    mu: numpy array of shape (n,) representing the mean of the posterior.
    sigma: numpy array of shape (n,) representing the standard deviation of the posterior.
    f_best: float, the best observed value so far.
    
    Returns:
    numpy array of shape (n,) with scores. Higher scores indicate higher probability
    of improving upon f_best.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability that f(x) < f_best is P(Z < z) where Z ~ N(0,1)
    scores = norm.cdf(z)
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

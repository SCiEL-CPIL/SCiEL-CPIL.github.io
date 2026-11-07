"""Cost-aware acquisition using Probability of Improvement (PI) with cost normalization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Probability of Improvement (PI).
    
    Args:
        mu: Array of GP means, shape (2000,)
        sigma: Array of GP standard deviations, shape (2000,)
        f_best: Scalar, lowest observed value.
        cost: Array of costs for each candidate, shape (2000,)
        
    Returns:
        Array of scores, shape (2000,), higher is better.
    """
    epsilon = 1e-9
    
    # Handle zero or near-zero sigma
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Z-score: how many standard deviations below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(y < f_best)
    # PI = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Ensure PI is in [0, 1] (numerical stability)
    pi = np.clip(pi, 0.0, 1.0)
    
    # Cost-aware score: PI divided by cost
    # This encourages finding improvements while being mindful of evaluation cost
    scores = pi / (cost + epsilon)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

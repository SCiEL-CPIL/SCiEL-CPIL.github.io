"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PoI) acquisition function.
    
    Args:
        mu: Mean of the GP posterior (shape (N,))
        sigma: Standard deviation of the GP posterior (shape (N,))
        f_best: The best observed value so far
        
    Returns:
        scores: Probability that the candidate is better than f_best (shape (N,))
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(y < f_best)
    # Since we are minimizing, we want mu to be lower than f_best.
    # The probability that a sample from N(mu, sigma^2) is less than f_best is Phi((f_best - mu)/sigma).
    scores = norm.cdf(z)
    
    # Ensure scores are finite and within [0, 1]
    scores = np.clip(scores, 0.0, 1.0)
    
    return scores
# EVOLVE-BLOCK-END

import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function for minimization.
    
    For minimization, the improvement at point x is:
        I(x) = max(0, f_best - f(x))
    
    The probability of improvement is:
        P(I(x) > 0) = P(f(x) < f_best)
                   = P((f(x) - mu(x)) / sigma(x) < (f_best - mu(x)) / sigma(x))
                   = Phi((f_best - mu(x)) / sigma(x))
    
    where Phi is the standard normal CDF.
    
    This aggressively exploits regions where the GP predicts values below f_best.
    """
    # Handle edge cases where sigma might be zero or very small
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized improvement: z = (f_best - mu) / sigma
    # Higher z means higher probability that f(x) < f_best
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement = Phi(z)
    # Using the standard normal CDF
    scores = norm.cdf(z)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Clip to [0, 1] to be safe (should already be in this range)
    scores = np.clip(scores, 0.0, 1.0)
    
    return scores

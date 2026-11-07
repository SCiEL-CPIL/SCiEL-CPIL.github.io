import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI)
    # For minimization, we want to maximize the probability that f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Compute CDF of standard normal distribution
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores

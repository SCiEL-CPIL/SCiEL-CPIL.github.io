import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI)
    # This is in the same family as EI but focuses on the probability
    # of finding a point better than f_best rather than the expected amount.
    # It is often more robust in early stages of optimization.
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Return the CDF of z
    # If mu < f_best and sigma is small, z is large positive, CDF -> 1
    # If mu > f_best, z is negative, CDF -> 0
    scores = norm.cdf(z)
    
    return scores

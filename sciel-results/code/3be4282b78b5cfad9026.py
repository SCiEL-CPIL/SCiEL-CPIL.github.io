import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PI) divided by cost.
    # PI is robust to the magnitude of f_best and encourages exploration
    # of regions with high probability of improvement, penalized by cost.
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware score
    score = pi / cost_safe
    
    return score

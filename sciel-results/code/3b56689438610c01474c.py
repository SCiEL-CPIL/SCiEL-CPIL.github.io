import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the Probability of Improvement (PI)
    # This is the probability that f(x) < f_best
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Ensure cost is at least 1 to avoid division by very small numbers
    # The problem states cost = 1 + mean(x_scaled), so cost >= 1
    cost_safe = np.maximum(cost, 1.0)
    
    # Cost-Effective Probability of Improvement (CEPI)
    score = pi / cost_safe
    
    # Replace any non-finite values with 0 to ensure validity
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

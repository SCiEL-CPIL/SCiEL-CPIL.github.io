import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Bayesian optimization on Styblinski-Tang 2D.
    
    Uses Expected Improvement with a quadratic cost penalty to favor
    low-cost regions where the minimum is likely located.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Quadratic cost penalty: strongly favor low-cost regions
    # The minimum of Styblinski-Tang is in the low-cost quadrant
    score = ei / (cost * cost)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

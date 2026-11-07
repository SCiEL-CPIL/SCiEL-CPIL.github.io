import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI (exploitation)
    ei_cost = ei / np.maximum(cost, eps)
    
    # Uncertainty bonus (exploration), also cost-normalized
    explore = sigma_safe / np.maximum(cost, eps)
    
    # Blend: mostly exploitation, some exploration
    score = 0.85 * ei_cost + 0.15 * explore
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

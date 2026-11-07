import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (due to numerical issues it might be slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Ensure cost is at least 1 to avoid division by very small numbers
    # The problem states cost = 1 + mean(x_scaled), so cost >= 1
    cost_safe = np.maximum(cost, 1.0)
    
    # Cost-Effective Expected Improvement (CEEI)
    score = ei / cost_safe
    
    # Replace any non-finite values with 0 to ensure validity
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

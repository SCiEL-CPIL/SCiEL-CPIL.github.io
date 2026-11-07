import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # f_best is the current best (lowest) value observed
    # We want to find points where mu is lower than f_best
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware adjustment
    # Cost is 1 + mean(x_scaled). 
    # Global minimum is at cost ~1.5.
    # Using a power less than 1 reduces the penalty for higher cost,
    # encouraging exploration of the more expensive but potentially better regions.
    cost_adjusted = np.power(cost, 0.5)
    
    # Score to maximize
    score = ei / cost_adjusted
    
    # Ensure the output is finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score

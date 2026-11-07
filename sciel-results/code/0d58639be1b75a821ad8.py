import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    p = norm.cdf(z)
    
    # Gain
    gain = f_best - mu
    
    # Probability-weighted improvement
    pw_gain = p * gain
    
    # Add a small exploration term to prevent stagnation
    # This ensures we still explore if all predictions are worse than best
    exploration = 0.1 * sigma_safe
    
    # Total numerator
    numerator = pw_gain + exploration
    
    # Score
    score = numerator / cost
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score

import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    xi = 0.0
    w = 0.5
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    
    # EI calculation
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus
    exploration = w * sigma_safe
    
    score = (ei + exploration) / cost_safe
    
    return score

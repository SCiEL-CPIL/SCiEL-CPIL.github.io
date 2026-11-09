import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI) for minimization
    
    # Safety clamp for sigma
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Z-score for improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal distribution
    
    # Use scipy for stable evaluation
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Clip to ensure non-negative EI (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Higher is better
    score = ei / cost
    
    return score

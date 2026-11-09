import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    Z = (f_best - mu) / sigma_safe
    
    # EI formula
    ei = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # Ensure EI is non-negative (it should be, but numerical errors might occur)
    ei = np.maximum(ei, 0)
    
    # Cost-aware score: maximize EI per unit cost
    score = ei / cost
    
    # Mask invalid values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score

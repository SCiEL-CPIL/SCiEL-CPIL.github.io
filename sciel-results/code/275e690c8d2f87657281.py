import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # Use a power > 1 to more aggressively prefer cheap evaluations
    cost_safe = np.maximum(cost, 1e-8)
    score = ei / (cost_safe ** 1.2)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

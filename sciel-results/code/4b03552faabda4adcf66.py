import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -10, 10)
    
    # Standard Normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (clip small numerical negatives)
    ei = np.maximum(ei, 0.0)
    
    # Add exploration bonus
    # Higher sigma means more exploration
    exploration_weight = 0.5
    score = ei + exploration_weight * sigma
    
    return score

import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1.0)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values in cdf/pdf
    z = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors)
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus
    # Add a term proportional to sigma to encourage exploration
    # This helps if EI is zero everywhere
    lambda_exp = 0.5
    acquisition_val = ei + lambda_exp * sigma
    
    # Cost adjustment
    # Divide by cost^gamma
    gamma = 1.0
    scores = acquisition_val / (cost ** gamma)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores

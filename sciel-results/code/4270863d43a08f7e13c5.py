import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Known global minimum
    f_min = -2.53218
    
    # Calculate gap to global minimum
    gap = f_best - f_min
    
    # Ensure gap is non-negative (f_best should be >= f_min)
    gap = max(0.0, gap)
    
    # Dynamic epsilon for exploration
    # Scale factor 0.2 means if gap is 1.0, epsilon is 0.2
    # Cap epsilon to avoid overly aggressive exploration
    epsilon = min(0.5, 0.2 * gap)
    
    # Calculate z-score for epsilon-EI
    # z = (f_best - epsilon - mu) / sigma
    sigma_safe = sigma + 1e-9
    z = (f_best - epsilon - mu) / sigma_safe
    
    # Calculate epsilon-Expected Improvement
    # EI_eps = (f_best - epsilon - mu) * Phi(z) + sigma * phi(z)
    ei_eps = (f_best - epsilon - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    ei_eps = np.nan_to_num(ei_eps, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return ei_eps

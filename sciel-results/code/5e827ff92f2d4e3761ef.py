import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Pure Expected Improvement (EI) for minimization
    # This is a structural change from the blended EI+UCB approach.
    # Pure EI adapts exploration naturally based on posterior uncertainty.
    
    # Handle sigma near zero to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score: z = (f_best - mu) / sigma
    # For minimization, improvement is positive when mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is always non-negative and naturally balances exploration/exploitation
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite and non-negative
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei

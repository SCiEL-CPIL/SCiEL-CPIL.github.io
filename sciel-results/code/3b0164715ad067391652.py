import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Use Expected Improvement (1D Hypervolume Improvement) for minimization
    # EI(x) = E[max(0, f_best - f(x))]
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.clip(sigma, 1e-8, None)
    cost_safe = np.clip(cost, 1e-8, None)
    
    # Initialize EI array
    ei = np.zeros_like(mu)
    
    # Handle cases where sigma is very small (deterministic limit)
    # If sigma -> 0, f(x) -> mu, so Improvement = max(0, f_best - mu)
    small_sigma_mask = sigma_safe < 1e-4
    
    if np.any(small_sigma_mask):
        ei[small_sigma_mask] = np.maximum(f_best - mu[small_sigma_mask], 0.0)
        
    # Handle cases with significant uncertainty
    if np.any(~small_sigma_mask):
        z = (f_best - mu[~small_sigma_mask]) / sigma_safe[~small_sigma_mask]
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)
        ei_calc = (f_best - mu[~small_sigma_mask]) * cdf + sigma_safe[~small_sigma_mask] * pdf
        ei[~small_sigma_mask] = np.maximum(ei_calc, 0.0)
        
    # Cost-aware score: EI / Cost
    scores = ei / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Aware Expected Improvement (EI) for minimization
    
    # Handle sigma = 0 case to avoid division by zero
    # If sigma is 0, the prediction is deterministic.
    # EI is max(0, f_best - mu)
    # We use a small epsilon to compute Z, then correct for sigma=0 if necessary
    # However, the limit of the EI formula as sigma->0 is max(0, f_best - mu)
    # So we can just use a safe sigma for Z calculation and the formula handles it well
    # provided we don't have numerical issues with Z -> inf.
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Z-score
    Z = (f_best - mu) / sigma_safe
    
    # Calculate EI components
    # Phi(Z) = CDF of standard normal
    # phi(Z) = PDF of standard normal
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    EI = (f_best - mu) * Phi_Z + sigma * phi_Z
    
    # Ensure EI is non-negative (numerical stability)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score: EI / cost
    # cost is shape (2000,) and >= 1
    score = EI / cost
    
    # Ensure finite scores
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

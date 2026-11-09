import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # This balances exploration (sigma) and exploitation (distance to f_best) adaptively.
    
    # Handle sigma == 0 to avoid division by zero
    eps = 1e-12
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # For points where sigma was effectively 0, the limit is max(f_best - mu, 0)
    mask = sigma <= eps
    if np.any(mask):
        ei[mask] = np.maximum(f_best - mu[mask], 0.0)
        
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei

import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    # Use a small epsilon for numerical stability
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using the original sigma in the second term is standard, 
    # but if sigma was 0, sigma_safe was used for z.
    # If sigma is 0, phi(z) is 0 (for large |z|), so the term vanishes.
    # If sigma is very small, phi(z) is small.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure finite values
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei

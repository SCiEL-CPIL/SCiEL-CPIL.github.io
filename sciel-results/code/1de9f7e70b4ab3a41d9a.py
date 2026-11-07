import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Use a small exploration parameter xi to balance exploitation and exploration
    # xi = 0 corresponds to standard EI, small positive xi favors exploration slightly
    xi = 0.01
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate the z-score for the expected improvement
    z = (f_best - mu - xi) / sigma_safe
    
    # Compute Expected Improvement
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all returned scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei

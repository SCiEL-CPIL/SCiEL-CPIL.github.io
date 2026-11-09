import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    """
    # Ensure sigma is not zero to avoid division by zero
    # Use a small epsilon
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical underflow/overflow in pdf/cdf
    # Values beyond +/- 10 are effectively 0 or 1 for CDF and 0 for PDF
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Compute PDF and CDF
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure scores are non-negative
    # Due to floating point arithmetic, EI might be slightly negative
    # when the improvement is negligible
    ei = np.maximum(ei, 0.0)
    
    return ei

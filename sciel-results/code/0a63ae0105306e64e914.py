import numpy as np
from scipy.special import erfc, exp

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    Uses scipy.special for numerical stability and speed.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # m = f_best - mu
    # If mu < f_best, m > 0 (potential improvement)
    m = f_best - mu
    s = sigma_safe
    
    # z = m / s
    z = m / s
    
    # Compute Phi(z) using erfc
    # Phi(z) = 0.5 * erfc(-z / sqrt(2))
    sqrt_2 = np.sqrt(2.0)
    phi_cdf = 0.5 * erfc(-z / sqrt_2)
    
    # Compute phi(z) (standard normal PDF)
    # phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    phi_pdf = exp(-0.5 * z * z) / sqrt_2pi
    
    # EI = m * Phi(z) + s * phi(z)
    ei = m * phi_cdf + s * phi_pdf
    
    # Ensure finite scores
    # If any numerical issues arise, replace non-finite with 0 or max
    # But erfc/exp should be stable.
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei

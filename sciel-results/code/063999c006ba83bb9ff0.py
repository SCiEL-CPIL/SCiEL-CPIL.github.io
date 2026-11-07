"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    Args:
        mu: (N,) array of posterior means.
        sigma: (N,) array of posterior standard deviations.
        f_best: float, best observed value so far.
        
    Returns:
        (N,) array of scores (higher is better).
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Improvement is defined as f_best - mu (since we minimize)
    improvement = f_best - mu
    
    # Standard normal Z-score
    z = improvement / sigma_safe
    
    # EI = improvement * Phi(z) + sigma * phi(z)
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # Clip negative EI to 0 (should not happen due to formula, but safe)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

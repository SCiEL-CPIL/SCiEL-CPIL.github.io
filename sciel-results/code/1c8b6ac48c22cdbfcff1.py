"""Rewritten program using Expected Improvement (EI) acquisition function."""
import numpy as np
from scipy.stats import norm

# EVOLVE-BLOCK-START
def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    Args:
        mu: Array of posterior means (shape 2000,).
        sigma: Array of posterior standard deviations (shape 2000,).
        f_best: The best (lowest) observed value so far.
        
    Returns:
        Array of scores (shape 2000,). Higher scores indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle sigma = 0 to avoid division by zero
    # If sigma is 0, the improvement is deterministic: max(f_best - mu, 0)
    # Otherwise, use the EI formula.
    
    # To avoid numerical issues, we can use a small epsilon or handle explicitly.
    # scipy.stats.norm.cdf and pdf handle 0 well, but division by 0 is bad.
    
    # Mask for sigma > 0
    mask = sigma > 1e-12
    
    scores = np.zeros_like(mu)
    
    # For points with negligible uncertainty
    scores[~mask] = np.maximum(f_best - mu[~mask], 0.0)
    
    # For points with uncertainty
    if np.any(mask):
        m = mu[mask]
        s = sigma[mask]
        
        # z = (f_best - mu) / sigma
        z = (f_best - m) / s
        
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        # Phi is norm.cdf, phi is norm.pdf
        ei = (f_best - m) * norm.cdf(z) + s * norm.pdf(z)
        
        scores[mask] = ei
    
    return scores
# EVOLVE-BLOCK-END

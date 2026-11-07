"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    Strategy:
    EI is the expected improvement over the current best value f_best.
    For a Gaussian posterior y ~ N(mu, sigma^2), the EI is:
    EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    Where:
    - Phi is the standard normal CDF
    - phi is the standard normal PDF
    
    We want to maximize EI, so higher scores are better.
    This function is deterministic and O(N) in complexity, making it very fast.
    It naturally balances exploration and exploitation:
    - When sigma is large, the term sigma * phi(z) encourages exploration.
    - When mu is low (close to 0) and f_best is high, the term (f_best - mu) * Phi(z) encourages exploitation.
    """
    n = len(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero in z calculation
    # sigma is standard deviation, should be >= 0. If 0, the point is certain.
    # If sigma is 0, z = inf (if f_best > mu) or -inf (if f_best < mu).
    # We handle the limit cases explicitly or via safe division.
    
    # Calculate improvement I = f_best - mu
    # If I <= 0, there is no expected improvement (since y ~ N(mu, sigma^2), 
    # if mu >= f_best, the probability of y < f_best is <= 0.5, but EI formula handles this? 
    # Actually, EI is defined as E[max(0, f_best - y)]. 
    # If mu >= f_best, the expected improvement is still positive if sigma > 0.
    # The formula EI = (f_best - mu)Phi(z) + sigma phi(z) works for all z.
    
    # Avoid division by zero for sigma
    sigma_safe = np.where(sigma > 1e-12, sigma, 1.0)
    
    # Standardized distance z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    # scipy.stats.norm.cdf and pdf are vectorized and fast
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # For points where sigma was 0 (or very small), the formula above with sigma_safe=1.0 
    # might be slightly off if we just plug in sigma=1.0 in the second term.
    # Let's refine:
    # If sigma is effectively 0:
    # If mu < f_best, EI = f_best - mu (deterministic improvement)
    # If mu >= f_best, EI = 0 (no improvement possible)
    
    # Mask for small sigma
    small_sigma_mask = sigma < 1e-12
    
    if np.any(small_sigma_mask):
        # Calculate EI for small sigma points
        # EI = max(0, f_best - mu)
        ei[small_sigma_mask] = np.maximum(0, f_best - mu[small_sigma_mask])
    
    # Ensure scores are finite
    # EI should always be non-negative and finite for finite inputs.
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # The task requires "higher meaning evaluate next".
    # EI is the expected improvement, so higher is better.
    return ei
# EVOLVE-BLOCK-END

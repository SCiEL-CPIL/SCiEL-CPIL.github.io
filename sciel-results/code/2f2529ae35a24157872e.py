"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function for minimization.
    
    Args:
        mu: Array of mean predictions (shape 2000,).
        sigma: Array of standard deviation predictions (shape 2000,).
        f_best: The current best (lowest) observed value.
        
    Returns:
        Array of PI scores (shape 2000,), higher is better.
    """
    # Ensure sigma is positive and finite to avoid division by zero
    # Using a small epsilon to handle the noise floor (1e-6) and numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Ensure mu is finite; replace non-finite with f_best (neutral probability 0.5 if sigma>0, but effectively 0 if we treat it as bad)
    # Actually, if mu is inf, z is -inf (if f_best is finite), cdf is 0. 
    # If mu is -inf, z is inf, cdf is 1.
    # We'll just let the math handle it but clip z to avoid extreme float issues.
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate Z-score for Probability of Improvement
    # We want P(f(x) < f_best)
    # f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu_safe) / sigma_safe
    
    # Clip z to avoid extreme values where norm.cdf might lose precision or return exactly 0/1
    # 50 is well within the range where CDF is distinguishable from 0 and 1 in double precision
    z_clipped = np.clip(z, -50, 50)
    
    # Compute PI
    pi = norm.cdf(z_clipped)
    
    # Ensure finite output
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    return pi
# EVOLVE-BLOCK-END

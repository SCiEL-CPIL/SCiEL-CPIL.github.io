"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with a dynamic
    improvement threshold (xi) that adapts to the current best value (f_best).
    
    Logic:
    - When f_best is large (far from optimum 0), we set xi to a small value
      (or zero) to encourage exploration and large jumps.
    - When f_best is small (close to optimum 0), we increase xi to require
      a more significant expected improvement, thus favoring exploitation
      and refining the solution.
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Dynamic xi calculation
    # We want xi to be small when f_best is large, and larger when f_best is small.
    # A simple mapping: xi = max(0, base_xi - f_best)
    # If f_best > base_xi, xi = 0 (pure EI, more exploratory relative to exploitation).
    # If f_best < base_xi, xi > 0 (stricter EI, more exploitative).
    
    # Tuning parameters:
    # base_xi: The maximum threshold we allow.
    # Weierstrass min is 0. Typical initial f_best might be ~0.1-0.5.
    # Let's set base_xi = 0.1.
    base_xi = 0.1
    
    xi = max(0.0, base_xi - f_best)
    
    # Calculate EI with dynamic xi
    # EI_xi = E[max(f_best - f(x) - xi, 0)]
    # z = (f_best - mu - xi) / sigma
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    
    z = (f_best - mu - xi) / sigma_safe
    
    # Handle numerical stability for large negative/positive z
    # norm.cdf and norm.pdf are generally stable, but we can clip if necessary.
    # For z < -10, cdf ~ 0, pdf ~ 0.
    # For z > 10, cdf ~ 1, pdf ~ 0.
    
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure scores are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END

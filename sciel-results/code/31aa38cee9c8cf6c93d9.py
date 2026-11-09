"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    EI(x) = E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu(x), sigma(x)^2)
    
    Analytical solution:
    If sigma(x) > 0:
        EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) 
               + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    If sigma(x) == 0:
        EI(x) = max(f_best - mu(x), 0)
    """
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite and non-negative
    # If mu > f_best significantly, EI should be 0. The formula handles this.
    # However, numerical noise might cause tiny negative values or NaNs.
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

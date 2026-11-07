"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Handle cases where sigma is very small to avoid division by zero
    # If sigma is near 0, the distribution is a point mass.
    # If mu < f_best, EI is f_best - mu.
    # If mu >= f_best, EI is 0.
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # EI formula
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # If sigma was effectively 0 and mu >= f_best, ei should be 0.
    # The formula above handles this naturally as z -> -inf, Phi -> 0, phi -> 0.
    # If mu < f_best and sigma -> 0, z -> +inf, Phi -> 1, phi -> 0.
    # ei -> (f_best - mu) * 1 + 0 = f_best - mu. Correct.
    
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

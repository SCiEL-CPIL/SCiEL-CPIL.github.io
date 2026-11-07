"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute Probability of Improvement (PI) acquisition scores.
    
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    Higher scores mean higher probability of improving upon f_best.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # PI = Phi(z)
    pi = norm.cdf(z)
    
    # Ensure finite output
    pi = np.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Clip to [0, 1] for safety
    pi = np.clip(pi, 0.0, 1.0)
    
    return pi
# EVOLVE-BLOCK-END

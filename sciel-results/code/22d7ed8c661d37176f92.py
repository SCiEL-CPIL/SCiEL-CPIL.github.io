"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
    where Z = (f_best - mu(x)) / sigma(x)
    
    Returns scores where higher values indicate better candidates.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Z
    Z = (f_best - mu) / sigma
    
    # Compute EI
    # norm.cdf(Z) is Phi(Z)
    # norm.pdf(Z) is phi(Z)
    EI = (f_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    return EI
# EVOLVE-BLOCK-END

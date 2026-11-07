"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    EI is a standard and robust choice for Bayesian optimization, 
    particularly in low-dimensional spaces with limited evaluations.
    """
    # Handle potential numerical issues with sigma
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    
    # z-score for the improvement
    z = (f_best - mu) / safe_sigma
    
    # Standard normal PDF and CDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite and non-negative (EI is non-negative by definition)
    # In case of numerical artifacts, clip to 0
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

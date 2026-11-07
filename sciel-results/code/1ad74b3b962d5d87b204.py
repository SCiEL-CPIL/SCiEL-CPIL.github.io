"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement (CA-EI)
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using scipy.stats.norm.cdf for Phi and .pdf for phi
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # Avoid division by zero if cost is 0 (though cost >= 1 here)
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

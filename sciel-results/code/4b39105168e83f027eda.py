"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Expected Improvement (CEI) for minimization.
    # This approach maximizes the expected improvement in the best observed value,
    # divided by the evaluation cost.
    
    # Avoid division by zero in sigma
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    
    # Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) - sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z - sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical errors can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Normalize by cost
    scores = ei / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

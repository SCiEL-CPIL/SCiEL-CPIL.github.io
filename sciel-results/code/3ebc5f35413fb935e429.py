"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI) with adaptive beta
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # We divide by cost to get a cost-aware score.
    
    # Safety: ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    # Use scipy for numerical stability
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is always non-negative (EI >= 0)
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost-aware normalization
    # Divide by cost to penalize expensive evaluations
    scores = EI / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Deterministic Cost-Penalized Expected Improvement (DCEI).
    
    Strategy:
    Use additive cost penalty instead of multiplicative division.
    Score = EI - lambda * cost
    
    Rationale:
    Multiplicative division (EI/cost) can be too aggressive in penalizing
    slightly more expensive points, potentially trapping the search in
    low-cost regions with mediocre improvement. Additive penalty allows
    high-improvement candidates to overcome moderate cost penalties,
    creating a clearer trade-off that may escape local optima.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple greedy exploitation if time is critical
        scores = -mu / cost
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in CDF/PDF
    z_clipped = np.clip(z, -30, 30)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI minus lambda * cost
    # lambda = 0.1 as suggested in directive
    lambda_cost = 0.1
    scores = ei - lambda_cost * cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

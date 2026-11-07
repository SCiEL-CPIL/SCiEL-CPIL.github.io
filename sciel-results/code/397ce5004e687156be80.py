"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    EI provides better exploitation than PI while still encouraging exploration
    through the sigma term. We use a smooth cost-aware weighting.
    """
    # Deadline guard: if we're close to the limit, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Phi(z) = CDF, phi(z) = PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = improvement * Phi(z) + sigma * phi(z)
    ei = improvement * cdf_z + sigma_safe * pdf_z
    
    # Clamp to non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: higher EI and lower cost = better
    # Cost is in [1, 2], so dividing by cost penalizes expensive points
    # Using a power to make cost penalty more pronounced
    score = ei / (cost ** 1.5)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
    # EVOLVE-BLOCK-END

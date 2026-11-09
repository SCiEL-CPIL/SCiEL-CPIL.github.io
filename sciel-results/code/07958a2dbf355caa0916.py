"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Combine EI and an exploration term (PI * sigma)
    # PI * sigma represents the expected improvement from exploring uncertain regions
    # where there is a high probability of finding a better solution
    exploration_term = 0.5 * pi * sigma_safe
    acquisition_value = ei + exploration_term
    
    # Penalize by cost
    # Direct division by cost is a standard cost-aware approach
    # Cost is in range [1, 2]
    score = acquisition_value / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with Uncertainty Bonus
    # EI for minimization
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Uncertainty bonus to encourage exploration
    # The coefficient 0.1 is a heuristic balance between exploitation (EI) and exploration (sigma)
    bonus = 0.1 * sigma_safe
    
    # Combine and normalize by cost
    score = (ei + bonus) / cost
    
    # Ensure finite values
    return np.where(np.isfinite(score), score, 0.0)
# EVOLVE-BLOCK-END

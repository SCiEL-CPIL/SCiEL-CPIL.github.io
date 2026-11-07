"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma and cost are positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # Since we are minimizing, EI is the expected amount by which f(x) < f_best
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware score: EI normalized by sqrt(cost)
    # This balances exploitation (EI) with cost efficiency, 
    # using a softer penalty than linear cost division.
    score = ei / np.sqrt(cost_safe)
    
    # Handle any potential numerical issues (NaNs, Infs)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END

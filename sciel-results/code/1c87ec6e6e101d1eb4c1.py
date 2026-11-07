"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PI) for minimization
    # This is a distinct approach from Expected Improvement (EI).
    # PI measures the probability that f(x) < f_best.
    
    # Handle numerical stability for sigma
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # For minimization, this is the CDF of the standard normal at z
    pi = norm.cdf(z)
    
    # Cost-aware score: Probability of improvement per unit cost
    # This encourages finding improvements while penalizing expensive evaluations
    scores = pi / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END

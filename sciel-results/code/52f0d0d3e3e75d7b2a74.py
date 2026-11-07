"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI) for minimization
    # This is a different approach family from UCB.
    # EI measures the expected reduction in the best known value.
    
    # Handle numerical stability for sigma
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # CDF and PDF of standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement formula for minimization
    # E[max(0, f_best - f(x))]
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Cost-aware score: Improvement per unit cost
    scores = ei / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END

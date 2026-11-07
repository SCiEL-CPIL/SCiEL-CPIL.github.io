"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Ensure sigma is positive to avoid division by zero issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # Phi is CDF, phi is PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle cases where sigma was artificially clipped (originally near 0)
    # If sigma is effectively 0, EI should be 0 if mu >= f_best, else (f_best - mu)
    # However, with the formula above, if sigma->0:
    # if mu < f_best, z -> +inf, cdf->1, pdf->0, ei -> f_best - mu
    # if mu > f_best, z -> -inf, cdf->0, pdf->0, ei -> 0
    # This behavior is correct.
    
    # Cost-awareness: Divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END

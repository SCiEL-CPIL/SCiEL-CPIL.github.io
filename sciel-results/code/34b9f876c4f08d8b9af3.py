import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return -mu / np.maximum(cost, 1e-6)

    # Sanitize inputs
    sig = np.maximum(sigma, 1e-8)
    cst = np.maximum(cost, 1e-6)

    # Compute Z-scores for expected improvement
    # z = (f_best - mu) / sigma
    z_vals = (f_best - mu) / sig
    
    # Clip z for numerical stability in normal distribution functions
    z_clipped = np.clip(z_vals, -10.0, 10.0)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF
    pdf_term = norm.pdf(z_clipped)
    cdf_term = norm.cdf(z_clipped)
    
    # Raw EI
    raw_ei = (f_best - mu) * cdf_term + sig * pdf_term
    
    # Ensure EI is non-negative (numerical safety)
    raw_ei = np.maximum(raw_ei, 0.0)
    
    # Cost-aware weighting
    # Use a softened cost penalty: 1 / (cost ** 0.8)
    # This is equivalent to dividing by cost**0.8
    cost_weight = 1.0 / np.power(cst, 0.8)
    
    # Final score
    scores = raw_ei * cost_weight
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

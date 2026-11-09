import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Adaptive exploration weight using logarithmic decay
    # This provides a smooth transition from high exploration early on to exploitation later
    # When f_best is large (e.g., 20-50), beta should be larger (~2-3)
    # When f_best is small (e.g., 0.1-1), beta should be smaller (~0.5-1)
    # Using log scaling: beta = base + scale * log(1 + f_best)
    # If f_best = 50, beta = 0.5 + 0.8 * log(51) ≈ 0.5 + 0.8 * 3.93 ≈ 3.64
    # If f_best = 1, beta = 0.5 + 0.8 * log(2) ≈ 0.5 + 0.8 * 0.693 ≈ 1.05
    # If f_best = 0.1, beta = 0.5 + 0.8 * log(1.1) ≈ 0.5 + 0.8 * 0.095 ≈ 0.576
    
    beta = 0.5 + 0.8 * np.log(1 + max(f_best, 0))
    
    # Score = EI + beta * sigma
    score = ei + beta * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score

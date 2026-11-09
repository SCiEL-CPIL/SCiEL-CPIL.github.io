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
    
    # Logarithmic decay for exploration weight
    # This provides a smooth transition from high exploration early to exploitation late
    # Using log1p to handle f_best near 0 gracefully
    beta = 0.1 + 1.5 / (1.0 + np.log1p(max(f_best, 0.0)))
    
    # Score = EI + beta * sigma
    score = ei + beta * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score

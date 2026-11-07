import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is CDF and phi is PDF of standard normal
    
    # Handle case where sigma is zero or very small
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # Use scipy.stats.norm for accuracy
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    scores = (f_best - mu) * Phi + sigma_safe * phi
    
    # Set scores to 0 where sigma was effectively zero (no uncertainty)
    # Actually, if sigma is 0, EI should be max(0, f_best - mu)
    # But with our formula: if sigma=0, z = inf (if f_best > mu) or -inf (if f_best < mu)
    # If f_best > mu: z = +inf, Phi = 1, phi = 0, so EI = f_best - mu
    # If f_best <= mu: z = -inf, Phi = 0, phi = 0, so EI = 0
    # This is correct behavior
    
    return scores

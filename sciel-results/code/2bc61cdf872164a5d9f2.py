import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Probability of Improvement (PI) for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    pi = norm.cdf(z)
    
    # Ensure finite
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    return pi

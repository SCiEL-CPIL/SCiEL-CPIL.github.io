import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    # Assuming f(x) ~ N(mu, sigma^2)
    
    # Handle numerical stability for very small sigma
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    d = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(d) + sigma * phi(d)
    ei = (f_best - mu) * norm.cdf(d) + sigma_safe * norm.pdf(d)
    
    return ei

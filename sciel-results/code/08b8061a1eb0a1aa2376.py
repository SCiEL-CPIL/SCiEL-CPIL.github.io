import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # f(x) ~ N(mu, sigma^2)
    
    # Avoid division by zero in sigma
    # If sigma is very small, EI is either 0 or (f_best - mu) if mu < f_best
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Z-score for the improvement
    # We want P(f(x) < f_best) and the expectation of the improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # EI formula
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure finite values
    # If sigma is extremely small and mu > f_best, EI should be ~0
    # If sigma is extremely small and mu < f_best, EI should be ~ f_best - mu
    # The formula handles this, but let's mask where sigma is effectively 0 to avoid numerical issues if any
    # Actually, with eps, it's safe.
    
    return ei

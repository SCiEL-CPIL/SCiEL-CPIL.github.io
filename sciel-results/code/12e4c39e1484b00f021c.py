import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) for minimization
    # PI = P(f(x) < f_best)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # PI is the CDF of the standard normal at z
    pi = norm.cdf(z)
    
    # Handle numerical edge cases
    # If sigma is effectively 0, the function value is mu.
    # If mu < f_best, PI should be 1.
    # If mu >= f_best, PI should be 0.
    # The CDF calculation above handles this naturally if z is large +/- infinity,
    # but floating point limits might cause issues.
    
    mask_small_sigma = sigma < 1e-6
    if np.any(mask_small_sigma):
        pi[mask_small_sigma] = np.where(mu[mask_small_sigma] < f_best, 1.0, 0.0)
        
    return pi

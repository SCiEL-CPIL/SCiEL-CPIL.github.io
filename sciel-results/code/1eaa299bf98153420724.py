import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is non-negative and not zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical underflow/overflow in pdf/cdf
    # norm.pdf(10) ~ 7e-24, norm.cdf(10) ~ 1.0
    # norm.pdf(-10) ~ 7e-24, norm.cdf(-10) ~ 7e-24
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0)
    
    # Cost-aware score
    # Using 1/cost is standard. 
    # We can try 1/(cost**1.1) for slightly more aggressive cost penalty.
    score = ei / cost
    
    return score

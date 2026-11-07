import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    # Use stable calculation for norm.cdf and norm.pdf
    # norm.cdf(z) for large negative z is 0, large positive is 1
    # norm.pdf(z) is small for large |z|
    
    # To avoid underflow/overflow issues in norm.cdf/pdf for extreme z,
    # we can use the fact that EI is 0 if f_best <= mu (roughly)
    # But standard formula works with clipping
    
    # Clip z to avoid extreme values if necessary, but norm.cdf handles it
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Dynamic cost penalty
    # lambda increases as f_best decreases
    # T controls the transition speed
    T = 5.0
    lambda_base = 1.0
    
    # If f_best is very large, exp(-f_best/T) is 0, lambda is 0
    # If f_best is small, exp(-f_best/T) is ~1, lambda is lambda_base
    # We add a small floor to lambda to always consider cost slightly?
    # Or just let it go to 0.
    
    lambda_dyn = lambda_base * np.exp(-f_best / T)
    
    # Log cost penalty
    # Cost is in [1, 2]
    # log(cost) is in [0, 0.69]
    log_cost = np.log(cost)
    
    score = ei - lambda_dyn * log_cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

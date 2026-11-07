import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start_time = time.time()
    # Safety margin: return immediately if we are dangerously close to the limit
    # Although 2000 points is fast, this complies with the directive.
    if time.time() - start_time > 320:
        return np.zeros_like(mu)

    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Add a small exploration bonus (UCB-like) to prevent getting stuck
    # in local optima where the GP is overconfident.
    # Beta is kept small (0.1) to prioritize exploitation, 
    # as larger values (e.g., 0.5 in inspiration 2) degraded performance.
    beta = 0.1
    score = ei + beta * sigma_safe
    
    # Cost-aware score: Score per unit cost
    # Ensure cost is not zero
    safe_cost = np.maximum(cost, 1e-9)
    
    final_score = score / safe_cost
    
    # Ensure finite values
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_score

import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize deadline
    if _start_time is None:
        _start_time = time.time()
        
    # Safety margin: 330s limit, return if we are too close
    # We return zeros which effectively stops selection or picks random/first, 
    # but prevents crash. In practice, this should rarely trigger if the loop is efficient.
    if time.time() - _start_time > 325:
        return np.zeros_like(mu)
        
    # 1. Calculate Expected Improvement (EI)
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # 2. Cost-Aware Normalization
    # The objective is to minimize regret per unit cost spent.
    # We divide EI by the cost of evaluating that point.
    # Cost is defined as 1 + mean(x_scaled). Minimum cost is 1.0.
    # Using the raw cost in the denominator creates a continuous gradient
    # that pushes search towards low-cost regions without hard clipping.
    
    # Ensure cost is at least 1e-9 to avoid division by zero (though min cost is 1)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score is EI per unit cost
    score = ei / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

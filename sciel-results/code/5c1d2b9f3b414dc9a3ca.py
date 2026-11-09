import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()
    
    # Ensure numerical stability
    sigma = np.clip(sigma, 1e-12, None)
    cost = np.clip(cost, 1e-12, None)
    
    # Expected Improvement (minimization)
    delta = f_best - mu
    z = delta / sigma
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = delta * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # Using division by cost to maximize expected improvement per unit cost.
    # This encourages finding improvements in cheaper regions.
    score = ei / cost
    
    # Add exploration bonus to encourage visiting high uncertainty areas
    # This helps escape local minima in the small budget scenario.
    # The bonus is also cost-aware.
    score += 0.1 * sigma / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

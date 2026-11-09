import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI)
    # This approach uses the probabilistic Expected Improvement metric
    # rather than the heuristic LCB, providing a more nuanced balance
    # between exploitation and exploration.
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-8)
    
    # Cost-aware score: Maximize expected improvement per unit cost
    scores = ei / cost_safe
    
    # Handle NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

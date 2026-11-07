import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Maximizes Expected Improvement (EI) per unit of cost.
    """
    t0 = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Avoid division by zero in score calculation
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Compute Score: EI per unit cost
    score = ei / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard (safety margin)
    elapsed = time.time() - t0
    if elapsed > 280:
        # Fallback: return a simple heuristic if we are running out of time
        # This is unlikely to be triggered for 2000 points
        return np.zeros_like(score)
        
    return score

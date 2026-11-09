import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function for minimization.
    
    1. Calculate Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
       where z = (f_best - mu) / sigma.
    2. Normalize by cost to account for expensive evaluations.
    3. Return the score (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # Phi is CDF, phi is PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (numerical errors can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    cost_safe = np.maximum(cost, 1e-8)
    
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

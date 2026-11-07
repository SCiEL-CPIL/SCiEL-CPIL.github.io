# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) normalized by cost.
    
    Score = EI / cost
    
    This is the standard Expected Improvement acquisition function for minimization,
    divided by the evaluation cost to balance exploration/exploitation against expense.
    """
    # Avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for improvement
    # For minimization, improvement is (f_best - mu)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # We want to maximize EI per unit cost
    score = ei / cost
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

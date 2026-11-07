import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    This function calculates the Expected Improvement for each candidate point
    and normalizes it by the evaluation cost to maximize efficiency.
    EI is generally more effective than PI in multimodal landscapes like Rastrigin
    because it weighs both the probability and the magnitude of improvement.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Deadline guard
    t_start = time.time()
    deadline = 300.0
    
    # Ensure sigma is positive to avoid division by zero
    # Add a small epsilon to prevent numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the standard normal CDF and phi is the standard normal PDF
    
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    ei_scores = (f_best - mu) * Phi + sigma_safe * phi
    
    # EI should be non-negative, but numerical errors might cause small negatives
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # Cost-aware score: Maximize EI per unit cost
    # We divide by cost to prioritize cheaper evaluations with high expected improvement
    cost_safe = np.maximum(cost, 1e-6)
    score = ei_scores / cost_safe
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    t_elapsed = time.time() - t_start
    if t_elapsed > deadline - 5.0:
        # Safety margin: return current result
        return score
    
    return score

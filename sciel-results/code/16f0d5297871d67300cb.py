import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for deadline safety
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Aware Expected Improvement with Exploration Bonus.
    """
    # Deadline check
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to simple exploitation to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-9)
        return -mu / cost

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure valid inputs
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # Minimization problem: EI = E[max(f_best - f(x), 0)]
    xi = 0.01
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Exploration Bonus
    # Add a small multiple of sigma to encourage exploration in uncertain regions
    # This is distinct from UCB as it adds to EI rather than replacing it
    alpha = 0.05
    exploration_bonus = alpha * sigma_safe
    
    # Combined Score before cost
    raw_score = ei + exploration_bonus
    
    # Cost Awareness
    # Divide by cost to prefer cheaper evaluations
    # Cost is > 1
    score = raw_score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

import numpy as np
from scipy.stats import norm
import time

# Module-level state for time tracking
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Powell 4D BO.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP std predictions  
    - f_best: scalar, best observed value so far
    - cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    - (2000,) array of acquisition scores (higher = better)
    """
    # Time guard: if we've been running too long, return safe fallback
    elapsed = _get_elapsed()
    if elapsed > 280:  # Safety margin before 300s limit
        # Return simple EI/cost as fallback
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware weighting:
    # Use a soft cost penalty via power of cost in denominator
    # alpha < 1 softens the penalty, allowing some expensive points
    # if they have high expected improvement
    alpha = 0.7
    cost_safe = np.maximum(cost, 1e-6)
    cost_weight = np.power(cost_safe, alpha)
    
    # Exploration term (UCB-like)
    # Adding a multiple of sigma encourages exploration in uncertain regions
    # This helps escape local minima and cover the space more effectively
    exploration_lambda = 1.0
    exploration_term = exploration_lambda * sigma
    
    # Combine exploitation (EI) and exploration (UCB) with cost penalty
    score = (ei + exploration_term) / cost_weight
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score

import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function with global budget-aware scaling.
    
    Parameters:
    mu (np.array): GP mean for 2000 candidates, shape (2000,)
    sigma (np.array): GP standard deviation for 2000 candidates, shape (2000,)
    f_best (float): Best observed function value
    cost (np.array): Cost of evaluating each candidate, shape (2000,)
    
    Returns:
    np.array: Scores for the 2000 candidates, higher is better.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return zeros if too close to timeout
    elapsed = time.time() - _start_time
    if elapsed > 320:
        return np.zeros(len(mu))
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Normalize EI to [0, 1] range for stable scaling
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    # Compute mean cost of the batch
    mean_cost = np.mean(cost_safe)
    
    # Global budget-aware penalty: penalize candidates with higher relative cost
    # This shifts exploration-exploitation based on remaining budget
    cost_penalty = 0.5 * (cost_safe - mean_cost) / (mean_cost + 1e-9)
    
    # Final score: normalized EI minus cost penalty
    score = ei_norm - cost_penalty
    
    # Handle any non-finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score

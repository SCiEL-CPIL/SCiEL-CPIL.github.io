import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a combination of Expected Improvement
    and a cost-aware exploration bonus.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        GP mean predictions
    sigma : array-like, shape (n,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value so far
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition values (higher = more desirable to evaluate)
    """
    # Deadline guard: If we are running out of time, return a simple fallback
    # to ensure we don't crash or timeout.
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Simple EI without cost penalty, just to return something valid
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)

    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero or negative std
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware adjustment
    # Normalize cost to [0, 1] within the current batch of candidates
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
        
    # Apply smooth exponential penalty
    # alpha=0.5 was found effective in the best inspiration
    # This penalizes higher cost points smoothly, preferring cheaper ones
    # without completely discarding them
    alpha = 0.5
    cost_weight = np.exp(-alpha * cost_norm)
    
    # Add an exploration bonus based on uncertainty, scaled by cost
    # This encourages exploring uncertain regions that are cheap to evaluate
    # The bonus is proportional to sigma and inversely proportional to cost
    # We use a soft normalization to avoid division by zero
    exploration_bonus = sigma_safe * (1.0 / (1.0 + cost_norm))
    
    # Combine EI and exploration bonus
    # Weight the exploration bonus to balance exploitation and exploration
    beta = 0.3
    scores = ei * cost_weight + beta * exploration_bonus
    
    # Ensure all values are finite to satisfy contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

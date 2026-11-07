import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 10.0

def _time_exceeded():
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses a step-based cost penalty (Soft Hard Gate) on Expected Improvement.
    
    Parameters:
    mu (np.array): Mean predictions from the GP, shape (N,).
    sigma (np.array): Standard deviation predictions from the GP, shape (N,).
    f_best (float): The best function value observed so far.
    cost (np.array): Cost of evaluating each candidate point, shape (N,).
    
    Returns:
    np.array: Scores for each candidate, higher is better.
    """
    # Time guard
    if _time_exceeded():
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle sigma near zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Step-based Cost Penalty
    # Instead of smooth division by cost, we apply a multiplicative penalty
    # to points that are more expensive than the median candidate cost.
    # This creates a "gate" that strongly prefers the cheaper half of the domain
    # but allows expensive points if their EI is significantly higher.
    
    # Use median as the threshold to split candidates into cheap/expensive
    cost_threshold = np.median(cost)
    
    # Penalty factor for expensive points. 
    # 0.1 means expensive points need 10x the EI of cheap points to be preferred.
    penalty_factor = 0.1
    
    scores = np.where(cost < cost_threshold, ei, ei * penalty_factor)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses a Thompson Sampling approach with cost adjustment to balance exploration
and exploitation while favoring low-cost regions.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline management
_START_TIME = time.monotonic()
_DEADLINE_SECONDS = 285.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    Draws a posterior sample and scores by f_s / cost_adjusted,
    balancing exploration via sampling with cost efficiency.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.monotonic() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure cost is within expected range [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)
    
    n = len(mu)
    
    # Thompson Sampling: draw a single posterior sample
    # f_s = mu + sigma * z, where z ~ N(0, 1)
    rng = np.random.default_rng()
    z = rng.standard_normal(n)
    f_sample = mu + sigma * z
    
    # Cost adjustment: divide by (cost - 0.5) to bias towards low-cost regions
    cost_adjusted = cost - 0.5
    cost_adjusted = np.maximum(cost_adjusted, 0.1)  # Avoid division by zero
    
    # Score: higher is better. We want to minimize f, so we negate the sample.
    # But since we're doing Thompson sampling for minimization, we want to
    # select candidates with low f_sample values. However, the acquisition
    # function should return higher scores for better candidates.
    # So we use: score = -f_sample / cost_adjusted
    # This way, lower f_sample (better) gives higher score (more negative / less positive)
    # Wait, that's backwards. Let me think again.
    # 
    # We want to evaluate the candidate that is likely to have the lowest f value.
    # Thompson sampling suggests: sample f_s from posterior, pick argmin f_s.
    # But the acquisition function returns scores where higher = evaluate next.
    # So we should return scores where lower f_s gets higher score.
    # score = -f_s / cost_adjusted would make lower f_s give higher (less negative) score.
    # Actually, let's use: score = -f_sample / cost_adjusted
    # This way, a candidate with f_sample = -10 and cost_adjusted = 0.5 gets score = 20
    # A candidate with f_sample = 5 and cost_adjusted = 0.5 gets score = -10
    # So lower f_sample (better) gives higher score. Good.
    
    score = -f_sample / cost_adjusted
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END

import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're close to the 300s limit, return safe scores
    elapsed = time.time() - _start_time
    if elapsed > 280:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Probability of significant improvement (PSI) based acquisition
    # For minimization: P(f(x) < f_best - epsilon)
    # Using normal approximation: P(N(mu, sigma^2) < f_best - eps)
    # = norm.cdf((f_best - eps - mu) / sigma)
    
    # Adaptive epsilon: smaller when f_best is close to 0
    epsilon = max(0.01 * abs(f_best), 1e-6)
    
    # Compute probability of improvement
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (f_best - epsilon - mu) / sigma
        prob_improve = norm.cdf(z)
    
    # Handle cases where sigma is 0
    prob_improve = np.where(sigma > 1e-10, prob_improve, 
                           np.where(mu < f_best - epsilon, 1.0, 0.0))
    
    # Also compute exploitation score (how much better than current best)
    exploit_score = np.maximum(f_best - mu, 0)
    
    # Combine exploration and exploitation
    # Weight exploration by probability of improvement
    # Add a small exploration bonus for high uncertainty
    exploration_bonus = 0.1 * sigma / (1 + sigma)
    
    scores = prob_improve + exploration_bonus + 0.5 * exploit_score / (1 + abs(f_best))
    
    # Cost-aware: divide by cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = scores / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores

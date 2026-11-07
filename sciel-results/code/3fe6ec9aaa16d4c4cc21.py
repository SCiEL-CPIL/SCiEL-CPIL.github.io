import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the 330s limit, return a fast, safe fallback
    if time.time() - _start_time > 325:
        # Simple exploitation: prefer low mu and low cost
        safe_cost = np.maximum(cost, 1e-6)
        return -mu / safe_cost

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)
    
    # Compute z-score for Probability of Improvement
    # For minimization, improvement is f_best - y
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in norm.cdf
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement (PI)
    # PI = P(y < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Ensure PI is within [0, 1]
    pi = np.clip(pi, 0.0, 1.0)
    
    # Cost-aware score: PI per unit cost
    # Using cost^0.8 based on inspiration 1's success, which softened the cost penalty
    # to allow exploration of higher-cost regions if the probability of improvement is significant.
    # This is a structural shift from EI (magnitude-focused) to PI (probability-focused),
    # which "thins" the search by prioritizing high-confidence improvements.
    cost_penalty = safe_cost ** 0.8
    score = pi / cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

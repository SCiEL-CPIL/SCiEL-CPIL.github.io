"""Rewritten program: PI acquisition with multiplicative cost penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track time for deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 320.0  # Safety margin before 330s limit

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the limit, return a fast fallback
    if _elapsed() > _DEADLINE - 5.0:
        # Fast fallback: PI with cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        score = pi / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Probability of Improvement (PI) for minimization
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Clip to valid probability range
    pi = np.clip(pi, 0.0, 1.0)
    
    # Cost-aware: use exponentiation for soft cost penalty
    # score = PI^(1/cost) - this naturally prioritizes cheap, high-confidence improvements
    # As cost increases, the exponent decreases, making the score less sensitive to PI
    # For cost=1, score = PI; for cost=2, score = sqrt(PI), etc.
    # This creates a natural balance: expensive points need high PI to be competitive
    
    # Use log-space for numerical stability
    # score = exp(log(PI) / cost)
    # Handle PI=0 case: log(0) = -inf, so score = 0
    log_pi = np.log(np.maximum(pi, 1e-15))
    log_score = log_pi / cost_safe
    score = np.exp(log_score)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

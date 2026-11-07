"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 10.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _start_time) - _MARGIN

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using multiplicative Probability of Improvement
    with adaptive sigma boost, divided by cost.
    
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    The multiplicative form PI * (1 + beta * sigma) couples exploitation and exploration.
    Dividing by cost makes it cost-aware.
    
    Adaptive beta: higher when f_best is far from optimum (more exploration needed),
    lower when close to optimum (more exploitation).
    """
    # Deadline guard
    if _remaining_time() < 1.0:
        # Fast fallback
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        beta = 0.3
        scores = pi * (1.0 + beta * sigma_safe)
        cost_safe = np.maximum(cost, 1e-8)
        scores = scores / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Probability of Improvement (PI)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Adaptive beta: estimate how far we are from convergence
    # f_best is the lowest observed value. For these functions, minimum is 0 or -3.86278
    # We don't know the exact minimum, but we can use the spread of mu to estimate uncertainty
    # If f_best is close to 0 (or -3.86), we're likely near the optimum
    
    # Use the median sigma as a proxy for remaining uncertainty
    med_sigma = np.median(sigma_safe)
    
    # Adaptive beta: start with more exploration when uncertainty is high
    # Scale beta between 0.1 and 0.5 based on med_sigma
    # When med_sigma is large relative to f_best, we need more exploration
    # When med_sigma is small, we can exploit more
    
    # Normalize: if f_best is close to 0, we're near optimum
    # Use a soft threshold
    f_abs = abs(f_best)
    # If f_abs is small (< 0.1), we're close to optimum, reduce beta
    # If f_abs is large, increase beta
    # Smooth transition
    beta_base = 0.3
    # Scale beta by how much uncertainty remains relative to the current best
    # If med_sigma >> f_abs, we have lots of uncertainty, explore more
    ratio = med_sigma / (f_abs + 1e-6)
    # Clip ratio to [0.5, 2.0] for stability
    ratio_clipped = np.clip(ratio, 0.5, 2.0)
    beta = beta_base * ratio_clipped
    
    # Cost-aware exploration: in expensive regions, dampen exploration
    # cost ranges from 1 (cheapest) to 2 (most expensive)
    # Normalize cost to [0, 1] for the penalty
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-8
    cost_norm = (cost - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    
    # Dampen exploration in expensive regions
    # exploration_damp = 1 - 0.3 * cost_norm (reduce exploration by up to 30% in expensive regions)
    exploration_damp = 1.0 - 0.25 * cost_norm
    
    scores = pi * (1.0 + beta * sigma_safe * exploration_damp)
    
    # Cost safety
    cost_safe = np.maximum(cost, 1e-8)
    
    # Cost-aware: divide by cost
    scores = scores / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

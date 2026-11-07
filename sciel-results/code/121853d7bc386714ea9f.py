"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're too close to the 330s limit, return safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback: just use -mu / cost (deterministic exploitation)
        safe_cost = np.maximum(cost, 1e-9)
        return -mu / safe_cost
    
    n = len(mu)
    
    # Thompson Sampling: draw one sample from the posterior Gaussian
    # sample(x) ~ N(mu(x), sigma(x)^2)
    # Score: we want to minimize, so higher score = lower sampled value
    # Cost-aware: score = -sampled_value / cost
    rng = np.random.default_rng()
    eps = rng.standard_normal(n)
    sampled = mu + sigma * eps
    
    # Ensure finite
    sampled = np.where(np.isfinite(sampled), sampled, mu)
    
    safe_cost = np.maximum(cost, 1e-9)
    score = -sampled / safe_cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

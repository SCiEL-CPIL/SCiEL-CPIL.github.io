import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _call_count
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)

    sigma = np.maximum(sigma, 1e-6)
    z = (f_best - mu) / sigma
    
    # Probability of Improvement (PI) for minimization:
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is a fundamentally different acquisition from EI:
    # PI rewards any region where there's a chance of beating f_best,
    # without weighting by how much improvement is expected.
    # This tends to be more balanced for multimodal functions.
    pi = norm.cdf(z)
    
    # UCB-like term for additional exploration
    # For minimization: we want low mu and high sigma
    # Score = -mu/sigma_scale + beta * sigma/sigma_scale
    # Normalize by typical scale of mu and sigma
    mu_scale = np.std(mu) + 1e-9
    sigma_scale = np.std(sigma) + 1e-9
    
    # Adaptive beta: start exploratory, become more exploitative
    # Use f_best as a proxy for progress: if f_best is still high, explore more
    # Rastrigin min is 0, max on this domain is around 40+
    # f_best starts around 10-20, should drop toward 0
    # Normalize: when f_best is high relative to range, we're early
    f_best_norm = min(max(f_best / 50.0, 0.0), 1.0)  # 0=found min, 1=still high
    # More exploration when f_best is high (early), less when low (late)
    beta = 0.5 + 1.5 * f_best_norm  # Range: 0.5 (exploit) to 2.0 (explore)
    
    ucb_norm = (-mu / mu_scale) + beta * (sigma / sigma_scale)
    
    # Hybrid: blend PI and normalized UCB
    # PI is in [0,1], ucb_norm is unbounded
    # Scale ucb_norm to be comparable
    ucb_min = np.min(ucb_norm)
    ucb_max = np.max(ucb_norm)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_scaled = (ucb_norm - ucb_min) / ucb_range  # Now in [0,1]
    
    # Blend weight: more PI early, more UCB later? Or vice versa?
    # PI is inherently balanced; UCB adds directed exploration
    # Let's use a fixed blend that favors PI slightly
    w_pi = 0.6
    w_ucb = 0.4
    
    scores = w_pi * pi + w_ucb * ucb_scaled
    
    # Cost-aware: divide by cost to prefer cheap points
    # cost = 1 + mean(x_scaled), so range is [1, 2]
    # This has a modest effect since cost varies by at most 2x
    scores = scores / (cost + 1e-9)
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores

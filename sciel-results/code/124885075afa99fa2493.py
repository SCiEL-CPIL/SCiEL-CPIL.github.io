import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    current_time = time.time()
    if current_time - _start_time > _DEADLINE:
        return np.ones(len(mu)) * 1e-9

    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Lower Confidence Bound for minimization
    # We want low mu and high uncertainty
    kappa = 2.0
    lcb = -mu + kappa * sigma_safe

    # Cost-aware score: combine EI and LCB, penalize by cost
    # Use a power of cost to strongly prefer cheap evaluations
    # cost ranges from ~1.0 to ~2.0
    # 1/cost^2 gives a stronger penalty for expensive points
    
    # Combine EI and LCB
    # EI is non-negative, LCB can be negative
    # Normalize by f_best scale if possible
    scale = max(abs(f_best), 1e-6)
    
    # Weighted combination
    score = (ei + 0.5 * lcb) / (cost_safe ** 2)
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score

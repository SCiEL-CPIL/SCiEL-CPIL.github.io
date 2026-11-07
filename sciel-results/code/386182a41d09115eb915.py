"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _MARGIN:
        # Fallback: simple cost-aware PI
        sigma_safe = np.maximum(np.asarray(sigma), 1e-9)
        z = (f_best - np.asarray(mu)) / sigma_safe
        return norm.cdf(z) / (np.asarray(cost) + 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Dynamic LCB term
    min_possible = -0.8
    gap_to_min = max(f_best - min_possible, 0.0)
    
    # Beta decreases as we get closer to optimum
    beta = 0.5 * np.exp(-gap_to_min / 0.3) + 0.1
    
    # LCB score for minimization: -mu + beta * sigma
    # Shifted to be relative to f_best: (f_best - mu) + beta * sigma
    lcb_score = (f_best - mu) + beta * sigma_safe

    # Variance penalty: stronger when close to optimum (small gap)
    # alpha increases as gap decreases
    alpha = 2.0 * (1.0 + 2.0 * np.exp(-gap_to_min / 0.2))
    var_penalty = 1.0 / (1.0 + alpha * sigma_safe * sigma_safe)

    # Combine EI and LCB
    # Weights: 0.5 EI, 0.5 LCB
    combined = 0.5 * ei + 0.5 * lcb_score
    
    # Apply variance penalty
    combined = combined * var_penalty
    
    # Cost normalization
    score = combined / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

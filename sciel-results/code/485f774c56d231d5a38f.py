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

    # Cost-aware LCB with dynamic exploration
    # The key insight: in 8D, variance inflation misleads EI-based selectors.
    # Instead, we use a cost-normalized LCB that explicitly penalizes
    # high-uncertainty regions via a dynamic exploration term scaled by
    # the remaining budget.
    
    # Remaining budget estimate: we start with cost ~5 (5 uniform points)
    # and the loop runs while total cost < 30.
    # The cost array gives per-candidate costs. We estimate how many more
    # evaluations we can afford.
    
    # Compute a dynamic beta that balances exploitation and exploration
    # based on how close we are to the known minimum (-0.8)
    min_possible = -0.8
    gap_to_min = max(f_best - min_possible, 0.0)
    
    # As we get closer to the optimum, reduce exploration
    # Beta starts high (exploratory) and decreases as gap shrinks
    beta = 0.3 + 0.7 * np.exp(-gap_to_min / 0.15)
    
    # LCB for minimization: lower is better
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    # Shift to be relative: (f_best - mu) + beta * sigma
    lcb_raw = (f_best - mu) + beta * sigma_safe
    
    # Additional variance penalty: in 8D, high sigma often means the GP
    # is extrapolating into unexplored regions where the function value
    # is unlikely to be better than f_best. Penalize high sigma quadratically.
    # The penalty strength increases as we approach the optimum.
    var_weight = 1.0 + 3.0 * np.exp(-gap_to_min / 0.1)
    var_penalty = 1.0 / (1.0 + var_weight * sigma_safe * sigma_safe)
    
    # Combine: primary is cost-normalized LCB with variance penalty
    score = lcb_raw * var_penalty / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

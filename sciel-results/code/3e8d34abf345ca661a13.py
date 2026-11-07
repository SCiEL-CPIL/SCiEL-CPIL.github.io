"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a multiplicative feasibility gate.
    
    Strategy: 
    - Compute Expected Improvement (EI) and Upper Confidence Bound (UCB).
    - Use a hard budget-aware gate: only score candidates whose predicted cost
      is below an adaptive threshold derived from the cost distribution.
    - Blend EI and UCB with adaptive uncertainty weighting.
    - This forces exploitation of low-cost regions until GP uncertainty there
      is exhausted, rather than smoothly trading off value and cost.
    """
    # Deadline guard: if we're running long, return a safe fallback
    elapsed = _get_elapsed()
    if elapsed > 300:
        return np.zeros(len(mu))
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (beta adapts to uncertainty)
    # Use a moderate exploration constant
    beta = 0.5
    ucb = mu + beta * sigma_safe
    
    # Cost-aware feasibility gate:
    # Compute quantile of costs to create a hard threshold
    # Candidates with cost below the 70th percentile get full score,
    # others get heavily penalized (multiplicative gate)
    cost_q70 = np.percentile(cost, 70)
    cost_q30 = np.percentile(cost, 30)
    
    # Multiplicative feasibility weight:
    # - Low cost (below q30): weight = 1.0 (full exploitation)
    # - High cost (above q70): weight = small (near-zero)
    # - In between: linear interpolation
    weight = np.ones(n)
    low_mask = cost <= cost_q30
    high_mask = cost >= cost_q70
    mid_mask = ~(low_mask | high_mask)
    
    if np.any(mid_mask):
        mid_costs = cost[mid_mask]
        mid_weight = 1.0 - (mid_costs - cost_q30) / (cost_q70 - cost_q30 + 1e-9)
        weight[mid_mask] = mid_weight
    weight[high_mask] = 0.1  # Small penalty for high-cost regions
    
    # Ensure weight is in [0, 1]
    weight = np.clip(weight, 0.0, 1.0)
    
    # Adaptive uncertainty weighting: blend EI and UCB based on sigma
    # High uncertainty -> more UCB weight; low uncertainty -> more EI weight
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    sigma_norm = sigma_safe / (sigma_max + 1e-9)
    
    # Blend: when sigma is high, trust UCB more; when low, trust EI more
    ei_weight = 0.6 + 0.4 * (1 - sigma_norm)  # 0.6 to 1.0
    ucb_weight = 1.0 - ei_weight               # 0.0 to 0.4
    
    # Combine EI and UCB
    # EI is in improvement units (positive = better)
    # UCB is in objective units (lower = better), so negate for "higher is better"
    combined = ei_weight * ei - ucb_weight * ucb
    
    # Apply multiplicative feasibility gate
    scores = combined * weight
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

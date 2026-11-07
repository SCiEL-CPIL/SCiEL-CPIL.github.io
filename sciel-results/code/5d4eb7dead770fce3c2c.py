"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: total wall-clock budget is ~330s.
# Track start time to ensure we return before timeout.
_START_TIME = time.time()
_DEADLINE = 325.0  # 5s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-entropy UCB acquisition: combines exploitation (-mu) with
    information-gain-per-cost (sigma/cost) and a cost-aware exploration bonus.
    
    Parameters:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP std dev predictions (positive)
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    # Deadline guard: if we're too close to timeout, return a safe default
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive kappa: high exploration early, low exploitation late
    f_best_safe = max(f_best, 1e-10)
    log_fbest = np.log(f_best_safe)
    log_fbest_min = np.log(0.01)
    log_fbest_max = np.log(10.0)
    log_fbest_clipped = np.clip(log_fbest, log_fbest_min, log_fbest_max)
    t = (log_fbest_clipped - log_fbest_min) / (log_fbest_max - log_fbest_min)
    
    # kappa from 0.3 (exploit) to 1.2 (explore)
    kappa = 0.3 + 0.9 * t
    
    # Core UCB for minimization
    ucb = -mu + kappa * sigma_safe
    
    # Cost-entropy heuristic:
    # 1. Information gain per cost: sigma / cost rewards uncertain cheap points
    # 2. Cost-aware scaling: normalize cost to [0,1] range for smoother penalty
    cost_safe = np.maximum(cost, 1.0)
    cost_norm = (cost_safe - 1.0) / 1.0  # in [0, 1] since cost in [1, 2]
    
    # Information-per-cost term: higher when sigma is high and cost is low
    info_per_cost = sigma_safe / cost_safe
    
    # Blend: primary UCB scaled by cost factor + secondary info-per-cost bonus
    # Use a smooth cost discount: exp(-alpha * cost_norm) penalizes expensive points
    # alpha controls strength of cost penalty
    alpha = 0.5
    cost_discount = np.exp(-alpha * cost_norm)
    
    # Combine: main term is UCB * cost_discount, plus a bonus for info-per-cost
    # The info-per-cost term helps when UCB values are similar (late game)
    # Scale info term so it doesn't dominate early
    info_weight = 0.3 * (1.0 - t)  # more weight early when exploration matters
    
    score = ucb * cost_discount + info_weight * info_per_cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    # prizes: (5, 100)
    # weights: (5, 100)
    
    epsilon = 1e-9
    capacity = 25.0
    
    # Per-constraint ratio: prizes[i,j] / weights[i,j]
    safe_weights = weights + epsilon
    per_constraint_ratio = prizes / safe_weights  # (5, 100)
    
    # Harmonic mean across constraints: n / sum(1/ratio_i)
    inv_ratio = 1.0 / per_constraint_ratio  # (5, 100)
    harmonic_mean_ratio = 1.0 / (np.mean(inv_ratio, axis=0) + epsilon)
    
    # Min-ratio: minimum per-constraint ratio (bottleneck constraint)
    min_ratio = np.min(per_constraint_ratio, axis=0)
    
    # Geometric slack factor: exp(mean(log(1 - w_i/capacity)))
    slack_factors = 1 - weights / capacity  # (5, 100)
    slack_factors = np.clip(slack_factors, 1e-9, 1)
    log_slack = np.log(slack_factors)
    geometric_slack = np.exp(np.mean(log_slack, axis=0))
    
    # Soft-min of per-constraint ratios using log-sum-exp (smooth surrogate for min)
    # soft_min(x) = -logsumexp(-x / T) * T
    T = 0.05  # lower temperature: sharper bottleneck penalty, closer to hard min
    neg_ratio_scaled = -per_constraint_ratio / T  # (5, 100)
    max_neg = neg_ratio_scaled.max(axis=0, keepdims=True)  # (1, 100)
    log_sum_exp = np.log(np.sum(np.exp(neg_ratio_scaled - max_neg), axis=0)) + max_neg.squeeze(0)
    soft_min_ratio = -log_sum_exp * T  # (100,)
    
    # Combine multiple efficiency measures:
    # - Harmonic mean as base (good balance)
    # - Soft-min to smooth the bottleneck penalty (instead of hard min)
    # - Geometric slack to reward light items
    
    # Use soft-min instead of hard min for smoother gradient in selection
    bottleneck_factor = np.power(soft_min_ratio + epsilon, 0.7)
    slack_factor = np.power(geometric_slack, 0.4)
    
    heuristic = harmonic_mean_ratio * bottleneck_factor * slack_factor
    
    # Apply power to sharpen selection
    heuristic = np.power(heuristic + epsilon, 1.8)
    
    # Ensure positivity
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Check for non-finite values
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    # Verify shape
    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)
    
    return heuristic
# EVOLVE-BLOCK-END

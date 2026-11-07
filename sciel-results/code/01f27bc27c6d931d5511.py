"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # Calculate total prize per item
    total_prize = np.sum(prizes, axis=0)
    
    # Avoid division by zero
    epsilon = 1e-9
    
    # Approach: Combine multiple efficiency measures
    # 1. Per-constraint prize/weight ratio
    # 2. Harmonic mean across constraints (penalizes bottleneck)
    # 3. Min-ratio (strongest penalty for worst constraint)
    # 4. Geometric mean of slack factors
    
    # Per-constraint ratio: prizes[i,j] / weights[i,j]
    safe_weights = weights + epsilon
    per_constraint_ratio = prizes / safe_weights  # (5, 100)
    
    # Harmonic mean across constraints: n / sum(1/ratio_i)
    # This is equivalent to: 1 / mean(1/ratio_i)
    inv_ratio = 1.0 / per_constraint_ratio  # (5, 100)
    harmonic_mean_ratio = 1.0 / (np.mean(inv_ratio, axis=0) + epsilon)
    
    # Min-ratio: minimum per-constraint ratio (bottleneck constraint)
    min_ratio = np.min(per_constraint_ratio, axis=0)
    
    # Geometric slack factor: exp(mean(log(1 - w_i/capacity)))
    capacity = 25
    slack_factors = 1 - weights / capacity  # (5, 100)
    slack_factors = np.clip(slack_factors, 1e-9, 1)
    log_slack = np.log(slack_factors)
    geometric_slack = np.exp(np.mean(log_slack, axis=0))
    
    # Combine: 
    # - Use harmonic mean as base (good balance)
    # - Multiply by min_ratio^0.5 to emphasize bottleneck
    # - Multiply by geometric_slack^0.3 to reward light items
    # - Apply power 1.5 to amplify differences
    
    base = harmonic_mean_ratio
    bottleneck_factor = np.power(min_ratio + epsilon, 0.5)
    slack_factor = np.power(geometric_slack, 0.3)
    
    heuristic = base * bottleneck_factor * slack_factor
    
    # Apply power to sharpen selection
    heuristic = np.power(heuristic + epsilon, 1.5)
    
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

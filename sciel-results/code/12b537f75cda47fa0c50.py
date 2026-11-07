"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for ACO of multi-constraint knapsack.
    
    prizes: (5, 100) uniform [0,1]
    weights: (5, 100) uniform [0,1]
    Returns: (100,) array, larger = more attractive.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    epsilon = 1e-9
    capacity = 25.0
    
    # Ensure inputs are numpy arrays of float
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape variations, though contract says (5, 100)
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    # If prizes is (1, 100) and weights is (5, 100), we need to broadcast or handle carefully.
    # The contract specifies both are (5, 100). 
    # If prizes is (5, 100), we use per-constraint ratios.
    
    # Calculate per-constraint ratios: prizes[i,j] / weights[i,j]
    # This assumes the 'prize' relevant to constraint i is prizes[i,j].
    safe_weights = weights + epsilon
    per_constraint_ratio = prizes / safe_weights  # (5, 100)
    
    # Harmonic mean across constraints: n / sum(1/ratio_i)
    # This is equivalent to 1 / mean(1/ratio_i)
    inv_ratio = 1.0 / per_constraint_ratio  # (5, 100)
    harmonic_mean_ratio = 1.0 / (np.mean(inv_ratio, axis=0) + epsilon)  # (100,)
    
    # Soft-min of per-constraint ratios using log-sum-exp (smooth surrogate for min)
    # soft_min(x) = -logsumexp(-x / T) * T
    # Lower T makes it sharper (closer to hard min).
    # Inspiration 1 used T=0.05.
    T = 0.05
    neg_ratio_scaled = -per_constraint_ratio / T  # (5, 100)
    # For numerical stability, subtract max before exp
    max_neg = neg_ratio_scaled.max(axis=0, keepdims=True)  # (1, 100)
    log_sum_exp = np.log(np.sum(np.exp(neg_ratio_scaled - max_neg), axis=0)) + max_neg.squeeze(0)
    soft_min_ratio = -log_sum_exp * T  # (100,)
    
    # Geometric slack factor: exp(mean(log(1 - w_i/capacity)))
    # Rewards items that are light across all constraints.
    slack_factors = 1 - weights / capacity  # (5, 100)
    slack_factors = np.clip(slack_factors, epsilon, 1)
    log_slack = np.log(slack_factors)
    geometric_slack = np.exp(np.mean(log_slack, axis=0))  # (100,)
    
    # Combine multiple efficiency measures:
    # - Harmonic mean as base (good balance of ratios)
    # - Soft-min to penalize the bottleneck constraint
    # - Geometric slack to reward light items
    
    # Exponents tuned from Inspiration 1 (best score)
    # Bottleneck factor: emphasize the soft-min
    bottleneck_factor = np.power(soft_min_ratio + epsilon, 0.7)
    # Slack factor: emphasize packing efficiency
    slack_factor = np.power(geometric_slack, 0.5) # Slightly higher than Insp 1's 0.4 to favor lighter items more
    
    heuristic = harmonic_mean_ratio * bottleneck_factor * slack_factor
    
    # Apply power to sharpen selection
    # Inspiration 1 used 1.8. We use 1.8 to maintain strong focus.
    heuristic = np.power(heuristic + epsilon, 1.8)
    
    # Ensure positivity and handle non-finite values
    heuristic = np.where(heuristic > 0, heuristic, epsilon)
    heuristic = np.where(np.isfinite(heuristic), heuristic, epsilon)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    # Verify shape
    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)
        
    return heuristic
# EVOLVE-BLOCK-END

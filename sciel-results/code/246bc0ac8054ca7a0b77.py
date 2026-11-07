"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Use a blend of geometric mean and minimum of per-constraint efficiency ratios
    in log-space, then apply rank-based normalization for robustness.
    
    The geometric mean captures overall efficiency, while the minimum captures
    the binding constraint. A 0.65/0.35 blend in log-space balances these.
    """
    eps = 1e-9
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    
    # Calculate ratios for each constraint: (5, 100)
    # ratio[i, j] = prize[i, j] / weight[i, j]
    ratios = prizes / (weights + eps)
    
    # Compute log of ratios (for log-space combination)
    # Clip to avoid log of very small values
    log_ratios = np.log(np.maximum(ratios, eps))
    
    # Geometric mean in log-space: mean of log ratios
    log_geo_mean = np.mean(log_ratios, axis=0)
    
    # Minimum in log-space: min of log ratios
    log_min = np.min(log_ratios, axis=0)
    
    # Blend in log-space: 0.65 weight on geometric mean, 0.35 on minimum
    w_geo = 0.65
    w_min = 0.35
    log_score = w_geo * log_geo_mean + w_min * log_min
    
    # Exponentiate to get raw scores
    raw_scores = np.exp(log_score)
    
    # Rank-based normalization for robustness
    ranks = np.argsort(np.argsort(raw_scores)).astype(float)
    ranks = ranks + 1.0  # Shift to 1..n
    scores = ranks / (ranks.mean() + eps)
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END

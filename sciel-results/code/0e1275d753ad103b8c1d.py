import numpy as np
import time

_DEADLINE = 330.0
_START_TIME = time.time()

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Strategy:
    Uses a Scarcity-Weighted Marginal Utility with soft-max bottleneck penalty.
    1. Per-constraint efficiency is computed as prize/weight.
    2. Constraint scarcity is measured by the ratio of top-k total weight to mean total weight,
       identifying which constraints are most likely to be binding.
    3. A soft-max (log-sum-exp) is used to create a smooth penalty for items that are
       inefficient in the tightest constraints, avoiding discontinuities.
    4. The final heuristic combines the weighted geometric mean with this soft penalty.
    
    Args:
        prizes: numpy array of shape (5, 100) with prizes for each constraint/item.
        weights: numpy array of shape (5, 100) with weights for each constraint/item.
        
    Returns:
        numpy array of shape (100,) with heuristic scores (larger is more attractive).
    """
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        # Fallback to simple geometric mean if running out of time
        p = np.clip(prizes, 1e-9, None)
        w = np.clip(weights, 1e-9, None)
        ratios = p / w
        log_ratios = np.log(ratios)
        mean_log_ratio = np.mean(log_ratios, axis=0)
        heuristic = np.exp(mean_log_ratio)
        heuristic = np.maximum(heuristic, 1e-9)
        mean_h = np.mean(heuristic)
        if mean_h > 0:
            heuristic = heuristic / mean_h
        return heuristic

    # Ensure inputs are float64 arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    eps = 1e-9
    
    # 1. Calculate per-constraint ratios: Prize / Weight
    safe_weights = np.clip(weights, eps, None)
    safe_prizes = np.clip(prizes, eps, None)
    
    ratios = safe_prizes / safe_weights  # Shape (5, 100)
    log_ratios = np.log(ratios)  # Shape (5, 100)
    
    # 2. Calculate scarcity weights for each constraint using top-k/mean ratio
    # For each constraint, compute the total weight of the top-k items (by weight) vs mean total
    n_items = weights.shape[1]
    k = max(1, n_items // 10)  # Top 10% of items by weight
    
    # Sort weights in each constraint descending
    sorted_weights = np.sort(weights, axis=1)[:, ::-1]  # Shape (5, 100)
    top_k_weights = np.sum(sorted_weights[:, :k], axis=1)  # Shape (5,)
    mean_total_weights = np.mean(np.sum(weights, axis=1))  # Scalar
    
    # Scarcity ratio: how much more weight is in the top-k items compared to average
    # Higher ratio means the constraint is more likely to be binding
    scarcity_ratios = top_k_weights / (mean_total_weights * k / n_items + eps)
    
    # Normalize scarcity weights
    scarcity_weights = scarcity_ratios
    sum_scarcity = np.sum(scarcity_weights)
    
    if sum_scarcity < eps:
        scarcity_weights = np.ones(5)
        sum_scarcity = 5.0
    
    # 3. Compute Weighted Geometric Mean (base heuristic)
    # log(geo_mean) = sum( w_c * log(ratio_c) ) / sum(w_c)
    weighted_log_sum = np.sum((scarcity_weights[:, np.newaxis] / sum_scarcity) * log_ratios, axis=0)
    base_heuristic = np.exp(weighted_log_sum)
    
    # 4. Soft-max bottleneck penalty
    # Identify the worst (minimum) log-ratio across constraints for each item
    # Use log-sum-exp for a smooth approximation to min
    # soft_min(x) = -log(sum(exp(-x)))
    neg_log_ratios = -log_ratios  # Shape (5, 100)
    max_neg = np.max(neg_log_ratios, axis=0, keepdims=True)  # Shape (1, 100)
    shifted = neg_log_ratios - max_neg
    log_sum_exp = max_neg + np.log(np.sum(np.exp(shifted), axis=0))
    soft_min_log_ratio = -log_sum_exp  # Shape (100,)
    
    # Penalty: items with low soft_min (i.e., very bad in at least one constraint) get penalized
    # Scale the penalty
    penalty_scale = 0.5
    penalty = np.exp(penalty_scale * soft_min_log_ratio)
    
    # Combine: heuristic = base * penalty
    heuristic = base_heuristic * penalty
    
    # 5. Ensure positivity and finiteness
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    # 6. Normalize to have mean 1
    mean_h = np.mean(heuristic)
    if mean_h > 0:
        heuristic = heuristic / mean_h
        
    return heuristic

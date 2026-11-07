"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Combines the structural idea of feasibility-aware density (residual capacity
    scaling) with the refinement machinery of soft-min and geometric mean
    aggregation from the inspirations.
    
    Strategy:
    1. Compute per-constraint prize-to-weight ratios.
    2. Compute residual capacity for each constraint (25 - sum of weights).
    3. Scale each ratio by a factor that rewards items efficient in constraints
       with abundant residual capacity (i.e., not just the tightest ones).
    4. Aggregate across constraints using a combination of geometric mean
       (overall balance) and soft-min (bottleneck awareness).
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    eps = 1e-9
    capacity = 25.0
    
    # Ensure positive weights to avoid division issues
    safe_weights = np.maximum(weights, eps)
    
    # Per-constraint efficiency ratios: prize / weight
    ratios = prizes / safe_weights  # shape (5, 100)
    
    # Compute residual capacity for each constraint
    # residual_c = capacity - sum_i(weights[c, i])
    total_weights_per_constraint = weights.sum(axis=1)  # shape (5,)
    residual_capacity = capacity - total_weights_per_constraint  # shape (5,)
    
    # Ensure residual capacity is positive (clamp to small positive value)
    residual_capacity = np.maximum(residual_capacity, eps)
    
    # Normalize residual capacity to [0, 1] range across constraints
    # This tells us which constraints are most "abundant"
    residual_max = np.max(residual_capacity)
    residual_min = np.min(residual_capacity)
    if residual_max - residual_min > eps:
        normalized_residual = (residual_capacity - residual_min) / (residual_max - residual_min)
    else:
        normalized_residual = np.ones_like(residual_capacity) * 0.5
    
    # Scale each per-constraint ratio by a factor that rewards items
    # efficient in constraints with abundant residual capacity.
    # We use a weighted combination: higher weight for constraints with
    # more residual capacity (they are less tight, so being efficient there
    # is more valuable for fitting more items).
    # scale_c = 1 + alpha * normalized_residual_c
    alpha = 1.0  # strength of residual-aware scaling
    scale_factors = 1.0 + alpha * normalized_residual  # shape (5,)
    
    # Apply scaling to ratios
    scaled_ratios = ratios * scale_factors[:, np.newaxis]  # shape (5, 100)
    
    # Geometric mean across constraints - penalizes items heavy in any constraint
    log_efficiency = np.log(np.maximum(scaled_ratios, eps))
    geo_mean_efficiency = np.exp(log_efficiency.mean(axis=0))
    
    # Soft-Min approximation to identify the bottleneck constraint efficiency
    # SoftMin(x) = -1/beta * log(sum(exp(-beta * x)))
    beta = 2.0
    
    # For numerical stability, subtract the max of (-beta * scaled_ratios) before exp
    v = -beta * scaled_ratios
    v_max = np.max(v, axis=0, keepdims=True)
    
    exp_v = np.exp(v - v_max)
    sum_exp = np.sum(exp_v, axis=0)
    
    log_sum_exp = v_max + np.log(sum_exp + eps)
    soft_min_ratios = -1.0 / beta * log_sum_exp
    
    # Combine geometric mean and soft-min
    # Geometric mean captures overall multi-constraint balance
    # Soft-min captures worst-case bottleneck
    # Weight them to balance both concerns
    scores = 0.5 * geo_mean_efficiency + 0.5 * soft_min_ratios
    
    # Ensure all values are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, eps)
    
    return scores
# EVOLVE-BLOCK-END

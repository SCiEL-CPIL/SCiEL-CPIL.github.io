"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Combines:
    1. Per-constraint efficiency ratio (prize/weight) aggregated via geometric mean
       to penalize items that are heavy in any single constraint (bottleneck awareness).
    2. Total prize to total weight ratio for overall value density.
    3. Bottleneck-aware normalization using the max weight across constraints.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    # Ensure positive weights to avoid division issues
    safe_weights = np.maximum(weights, 1e-9)
    
    # Per-constraint efficiency: prize / weight for each constraint
    efficiency = prizes / safe_weights  # shape (5, 100)
    
    # Geometric mean across constraints - penalizes items heavy in any constraint
    # Use log-space for numerical stability
    log_efficiency = np.log(np.maximum(efficiency, 1e-9))
    geo_mean_efficiency = np.exp(log_efficiency.mean(axis=0))
    
    # Total prize to total weight ratio (overall value density)
    total_prize = prizes.sum(axis=0)
    total_weight = weights.sum(axis=0)
    total_ratio = total_prize / (total_weight + 1e-9)
    
    # Bottleneck-aware score: use max weight as the binding constraint indicator
    w_max = weights.max(axis=0)
    relative_cost = w_max / 25.0
    bottleneck_ratio = total_prize / (relative_cost + 1e-9)
    
    # Combine three signals:
    # - Geometric mean efficiency captures multi-constraint balance
    # - Total ratio captures overall value density
    # - Bottleneck ratio captures the most constraining dimension
    # Weight geometric mean most heavily since fitting all constraints is critical
    scores = 0.5 * geo_mean_efficiency + 0.2 * total_ratio + 0.3 * bottleneck_ratio
    
    # Ensure all values are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END

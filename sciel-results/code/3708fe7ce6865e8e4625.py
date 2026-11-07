"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
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
    # Use log-space to compute geometric mean
    log_efficiency = np.log(np.maximum(efficiency, 1e-9))
    geo_mean_efficiency = np.exp(log_efficiency.mean(axis=0))
    
    # Also compute total prize to weight ratio as a secondary signal
    total_prize = prizes.sum(axis=0)
    total_weight = weights.sum(axis=0)
    total_ratio = total_prize / (total_weight + 1e-9)
    
    # Combine: geometric mean captures multi-constraint balance, 
    # total ratio captures overall value density
    # Weight geometric mean more heavily since multi-constraint fit is critical
    scores = 0.7 * geo_mean_efficiency + 0.3 * total_ratio
    
    # Ensure all values are positive
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END

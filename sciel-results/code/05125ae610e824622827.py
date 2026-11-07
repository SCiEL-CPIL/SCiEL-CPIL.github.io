"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes and weights are shape (5, 100)
    
    # Per-constraint efficiency: prize / weight for each constraint
    # This captures the "density" along each individual constraint
    safe_weights = np.maximum(weights, 1e-9)
    efficiency = prizes / safe_weights  # shape (5, 100)
    
    # Geometric mean across constraints - penalizes items that are heavy
    # in any single constraint (bottleneck awareness)
    log_efficiency = np.log(np.maximum(efficiency, 1e-9))
    geo_mean_efficiency = np.exp(log_efficiency.mean(axis=0))
    
    # Total prize to total weight ratio (overall value density)
    total_prize = prizes.sum(axis=0)
    total_weight = weights.sum(axis=0)
    total_ratio = total_prize / (total_weight + 1e-9)
    
    # Combine geometric mean (multi-constraint balance) with total ratio
    # Weight geometric mean more heavily since fitting all constraints is critical
    scores = 0.7 * geo_mean_efficiency + 0.3 * total_ratio
    
    # Ensure all values are positive
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END

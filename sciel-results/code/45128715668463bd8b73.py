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
    n_constraints, n_items = prizes.shape
    
    # Safe weights to avoid division by zero
    safe_weights = np.maximum(weights, 1e-9)
    
    # Per-constraint efficiency: prize / weight for each constraint
    efficiency = prizes / safe_weights  # shape (5, 100)
    
    # Rank-based normalization per constraint: convert to [0, 1] scale
    # This makes the heuristic invariant to absolute scale
    # Use argsort of argsort to get ranks (0 = worst, n-1 = best)
    ranks = np.empty_like(efficiency, dtype=np.float64)
    for c in range(n_constraints):
        order = np.argsort(efficiency[c])
        ranks[c] = np.empty(n_items, dtype=np.float64)
        ranks[c][order] = np.arange(n_items, dtype=np.float64)
    
    # Normalize ranks to [0, 1]
    norm_ranks = ranks / (n_items - 1)  # shape (5, 100)
    
    # Geometric mean of normalized ranks across constraints
    # This penalizes items that are ranked poorly in any constraint
    # Use log-space for numerical stability
    log_norm_ranks = np.log(np.maximum(norm_ranks, 1e-9))
    geo_mean_rank = np.exp(log_norm_ranks.mean(axis=0))  # shape (100,)
    
    # Also compute the arithmetic mean of normalized ranks as a complement
    arith_mean_rank = norm_ranks.mean(axis=0)  # shape (100,)
    
    # Combine: geometric mean emphasizes balance, arithmetic mean emphasizes overall quality
    # Weight geometric mean more heavily since multi-constraint fit is critical
    scores = 0.6 * geo_mean_rank + 0.4 * arith_mean_rank
    
    # Ensure all values are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END

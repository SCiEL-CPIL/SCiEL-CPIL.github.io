"""New program for multi-constraint knapsack heuristic scoring."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using multi-criteria ranking.
    
    Approach:
    1. Normalize each constraint's weights independently to [0, 1].
    2. For each item, compute the maximum normalized weight across constraints (bottleneck).
    3. Compute a total prize score per item (sum of prizes across constraints).
    4. Normalize prize scores to [0, 1].
    5. Heuristic = normalized_prize * (1 / (max_norm_weight + epsilon)).
       This prioritizes items that are highly valuable relative to their worst-case resource consumption.
    
    Parameters:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,), one attractiveness score per item
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Step 1: Normalize each constraint's weights to [0, 1]
    w_min = weights.min(axis=1, keepdims=True)
    w_max = weights.max(axis=1, keepdims=True)
    w_range = w_max - w_min
    w_range = np.where(w_range > 1e-15, w_range, 1.0)
    norm_weights = (weights - w_min) / w_range  # shape (5, 100)
    
    # Step 2: Max normalized weight per item (worst-case resource consumption)
    max_norm_w = norm_weights.max(axis=0)  # shape (100,)
    
    # Step 3: Prize score per item using sum of prizes
    # Assuming the total prize is the sum of all prize components
    total_prizes = prizes.sum(axis=0)  # shape (100,)
    
    # Step 4: Normalize prize scores to [0, 1]
    p_min = total_prizes.min()
    p_max = total_prizes.max()
    p_range = p_max - p_min
    if p_range > 1e-15:
        norm_prize = (total_prizes - p_min) / p_range
    else:
        norm_prize = np.ones(100, dtype=np.float64)
    
    # Step 5: Heuristic = norm_prize * (1 / (max_norm_w + eps))
    eps = 1e-6
    # Inverse of max normalized weight
    inv_weight = 1.0 / (max_norm_w + eps)
    
    heuristic = norm_prize * inv_weight
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Scale to reasonable range (optional, but helps with pheromone balance)
    # The directive doesn't specify scaling, but keeping values in a similar range
    # to the previous program (max 1.0) is a safe bet for the ACO algorithm.
    h_max = heuristic.max()
    if h_max > 0 and np.isfinite(h_max):
        heuristic = heuristic / h_max
    
    return heuristic
# EVOLVE-BLOCK-END

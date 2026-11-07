"""New program for multi-constraint knapsack heuristic scoring."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using the specified multi-criteria ranking model.
    
    Model:
    1. Aggregate item value: Sum of prizes across constraints.
    2. Normalize item value to [0, 1] (Min-Max).
    3. Normalize each constraint's weights to [0, 1] (Min-Max).
    4. Identify worst-case resource consumption: Max normalized weight per item.
    5. Heuristic = Normalized Value * (1 / (Worst-Case Weight + epsilon)).
    
    Parameters:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,), one attractiveness score per item
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Step 1: Aggregate item value (Sum of prizes across constraints)
    # This represents the total potential value of selecting the item.
    item_values = prizes.sum(axis=0)  # shape (100,)
    
    # Step 2: Normalize item values to [0, 1]
    v_min = item_values.min()
    v_max = item_values.max()
    v_range = v_max - v_min
    if v_range > 1e-15:
        norm_value = (item_values - v_min) / v_range
    else:
        norm_value = np.ones(100, dtype=np.float64)
        
    # Step 3: Normalize each constraint's weights to [0, 1]
    w_min = weights.min(axis=1, keepdims=True)
    w_max = weights.max(axis=1, keepdims=True)
    w_range = w_max - w_min
    w_range = np.where(w_range > 1e-15, w_range, 1.0)
    norm_weights = (weights - w_min) / w_range  # shape (5, 100)
    
    # Step 4: Worst-case resource consumption (Max normalized weight)
    max_norm_w = norm_weights.max(axis=0)  # shape (100,)
    
    # Step 5: Heuristic = Norm Value * Inverse(Max Norm Weight)
    eps = 1e-6
    # Using a small epsilon to avoid division by zero for "free" items
    # Items with 0 normalized weight in all constraints get high score.
    inverse_w = 1.0 / (max_norm_w + eps)
    
    heuristic = norm_value * inverse_w
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Scale to reasonable range (Max 1.0) to match typical heuristic scales
    h_max = heuristic.max()
    if h_max > 0 and np.isfinite(h_max):
        heuristic = heuristic / h_max
        
    return heuristic
# EVOLVE-BLOCK-END

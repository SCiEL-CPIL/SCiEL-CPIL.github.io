import numpy as np
import time

def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    # Ensure both have the same number of rows
    if prizes.shape[0] == 1 and weights.shape[0] > 1:
        prizes = np.repeat(prizes, weights.shape[0], axis=0)
    elif weights.shape[0] == 1 and prizes.shape[0] > 1:
        weights = np.repeat(weights, prizes.shape[0], axis=0)
    
    num_constraints, num_items = prizes.shape
    
    eps = 1e-9
    
    # Normalize weights per constraint to [0, 1]
    w_min = np.min(weights, axis=1, keepdims=True)
    w_max = np.max(weights, axis=1, keepdims=True)
    w_range = w_max - w_min
    w_range = np.where(w_range < 1e-12, 1.0, w_range)
    weights_norm = (weights - w_min) / w_range  # shape (num_constraints, num_items)
    
    # Normalize prizes per constraint to [0, 1]
    p_min = np.min(prizes, axis=1, keepdims=True)
    p_max = np.max(prizes, axis=1, keepdims=True)
    p_range = p_max - p_min
    p_range = np.where(p_range < 1e-12, 1.0, p_range)
    prizes_norm = (prizes - p_min) / p_range  # shape (num_constraints, num_items)
    
    # For each item, find the bottleneck constraint (max normalized weight)
    bottleneck_idx = np.argmax(weights_norm, axis=0)  # shape (num_items,)
    
    # Get the prize and weight at the bottleneck constraint for each item
    item_indices = np.arange(num_items)
    bottleneck_prize = prizes_norm[item_indices, bottleneck_idx]  # shape (num_items,)
    bottleneck_weight = weights_norm[item_indices, bottleneck_idx]  # shape (num_items,)
    
    # Also compute a robust aggregate: sum of normalized prizes (total value)
    total_prize_norm = np.sum(prizes_norm, axis=0)  # shape (num_items,)
    
    # Heuristic: combine bottleneck efficiency with total value
    # Primary signal: prize at bottleneck / weight at bottleneck
    bottleneck_efficiency = bottleneck_prize / (bottleneck_weight + eps)
    
    # Secondary signal: total normalized prize / max normalized weight
    max_norm_weight = np.max(weights_norm, axis=0)
    total_value_efficiency = total_prize_norm / (max_norm_weight + eps)
    
    # Combine: geometric mean of the two efficiency signals
    # This balances local (bottleneck) and global (total) value considerations
    log_bottleneck = np.log(np.maximum(bottleneck_efficiency, eps))
    log_total = np.log(np.maximum(total_value_efficiency, eps))
    combined_log = 0.5 * (log_bottleneck + log_total)
    heuristic = np.exp(combined_log)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic

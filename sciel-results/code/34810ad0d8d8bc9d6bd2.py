import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes heuristic values for items in a multiple-constraint knapsack problem.
    
    Args:
        prizes: numpy array of shape (5, 100) with prize values.
        weights: numpy array of shape (5, 100) with weight values.
        
    Returns:
        numpy array of shape (100,) with heuristic values.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit
    
    # Ensure inputs are numpy arrays of float
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_constraints, n_items = weights.shape
    eps = 1e-9
    
    # Compute per-constraint residual capacity: capacity - sum of all weights in that dimension
    capacity = np.full(n_constraints, 25.0)
    total_weights_per_constraint = np.sum(weights, axis=1)  # shape: (5,)
    residual_capacity = capacity - total_weights_per_constraint  # shape: (5,)
    residual_capacity = np.maximum(residual_capacity, eps)  # avoid division by zero/negative
    
    # Per-constraint efficiency: prize / weight, shape: (5, 100)
    efficiencies = prizes / (weights + eps)
    
    # Scale each constraint's efficiency by its residual capacity
    # This makes items that are efficient in abundant resources more attractive
    scaled_efficiencies = efficiencies * residual_capacity[:, np.newaxis]
    
    # Use a soft-min (via log-sum-exp on negatives) to robustly combine constraints
    # This is a smooth surrogate for the min, emphasizing the bottleneck constraint
    # while still incorporating all constraints
    # log-sum-exp trick for numerical stability
    scaled_eff_log = np.log(scaled_efficiencies + eps)
    
    # Use a temperature parameter to control softness (smaller temp = closer to min)
    temp = 0.1
    scaled_eff_log_scaled = scaled_eff_log / temp
    
    # Find max for stability
    max_log = np.max(scaled_eff_log_scaled, axis=0)
    
    # log-sum-exp: max + log(sum(exp(x - max)))
    log_sum_exp = max_log + np.log(np.sum(np.exp(scaled_eff_log_scaled - max_log), axis=0))
    
    # The soft-min is -log-sum-exp(-x)/n, but we want a positive score
    # Actually, we want to penalize items where ANY constraint is tight
    # So we use: exp(-log_sum_exp(-scaled_eff_log)/n) or similar
    # Let's use: soft_min = exp(mean(log(exp(x)))) but that's just geometric mean
    # Instead, use a different approach: the harmonic mean of scaled efficiencies
    # which already emphasizes bottlenecks
    
    # Harmonic mean of scaled efficiencies
    reciprocal_scaled = 1.0 / (scaled_efficiencies + eps)
    avg_reciprocal = np.mean(reciprocal_scaled, axis=0)
    heuristic = 1.0 / (avg_reciprocal + eps)
    
    # Alternative: use a weighted geometric mean with weights based on residual capacity
    # Items that are efficient in constraints with more remaining capacity should be preferred
    log_heuristic = np.mean(np.log(scaled_efficiencies + eps), axis=0)
    heuristic_geom = np.exp(log_heuristic)
    
    # Combine harmonic mean and geometric mean
    # Harmonic mean is more robust to outliers (bad in one constraint)
    # Geometric mean gives more weight to all constraints equally
    heuristic_combined = 0.7 * heuristic + 0.3 * heuristic_geom
    
    # Normalize to [0, 1] range for better pheromone interaction
    max_h = np.max(heuristic_combined)
    min_h = np.min(heuristic_combined)
    if max_h > min_h:
        heuristic_normalized = (heuristic_combined - min_h) / (max_h - min_h)
    else:
        heuristic_normalized = np.ones_like(heuristic_combined)
    
    # Scale to reasonable range
    heuristic_final = 0.1 + heuristic_normalized * 0.9
    
    # Ensure values are finite and positive
    heuristic_final = np.where(np.isfinite(heuristic_final), heuristic_final, eps)
    heuristic_final = np.maximum(heuristic_final, eps)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic_final
    
    return heuristic_final

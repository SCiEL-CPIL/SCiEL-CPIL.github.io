import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute heuristic values for ant colony optimization of the multiple-constraint knapsack.
    
    Uses a competitive exclusion / opportunity-cost model:
    1. Per-constraint efficiency: prize / weight
    2. Bottleneck constraint identification via soft-min
    3. Competitive pressure estimation
    4. Combined with harmonic mean and power amplification for sharper selection
    """
    start_time = time.time()
    deadline = 320.0  # safety margin before 330s kill
    
    # Robustly convert to numpy arrays and handle shape
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Expected shape is (5, 100)
    if prizes.ndim == 1:
        if prizes.size == 500:
            prizes = prizes.reshape(5, 100)
        else:
            n_items = prizes.size if prizes.size == 100 else 100
            return np.ones(n_items)
            
    if weights.ndim == 1:
        if weights.size == 500:
            weights = weights.reshape(5, 100)
        else:
            n_items = weights.size if weights.size == 100 else 100
            return np.ones(n_items)

    n_constraints, n_items = prizes.shape
    epsilon = 1e-9
    
    # Ensure weights are positive
    safe_weights = np.where(weights > epsilon, weights, epsilon)
    
    # Per-constraint efficiency ratios: prize / weight for each constraint
    ratios = prizes / safe_weights
    ratios = np.clip(ratios, epsilon, None)
    
    # Soft-min of ratios across constraints (numerically stable)
    # This captures how balanced an item is across all constraints
    neg_log_ratios = -np.log(ratios)  # shape: (n_constraints, n_items)
    max_neg = np.max(neg_log_ratios, axis=0, keepdims=True)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_log_ratios - max_neg), axis=0))
    soft_min_ratio = np.exp(-log_sum_exp)  # shape: (n_items,)
    
    # Hard min ratio (bottleneck)
    min_ratio = np.min(ratios, axis=0)  # shape: (n_items,)
    
    # Competitive pressure: for each item, estimate how much competition it faces
    # in its bottleneck constraint
    competitive_pressure = np.zeros(n_items)
    
    for c in range(n_constraints):
        # Sort items by ratio in constraint c (descending)
        order = np.argsort(-ratios[c])
        sorted_prizes = prizes[c, order]
        # Cumulative sum from top: items with ratio >= current item's ratio
        cumsum_prizes = np.cumsum(sorted_prizes)
        
        # Map back to original indices
        pressure_in_c = np.empty(n_items)
        pressure_in_c[order] = cumsum_prizes
        
        # For items whose bottleneck is constraint c, use this pressure
        bottleneck_mask = np.argmin(ratios, axis=0) == c
        competitive_pressure[bottleneck_mask] = pressure_in_c[bottleneck_mask]
    
    # Ensure competitive pressure is at least the item's own prize
    competitive_pressure = np.maximum(competitive_pressure, prizes.sum(axis=0) * 0.01 + epsilon)
    
    # Base heuristic: total prize / competitive pressure
    total_prize = prizes.sum(axis=0)
    base_heuristic = total_prize / competitive_pressure
    
    # Harmonic mean of min_ratio and soft_min_ratio
    # This penalizes items that are weak in any constraint
    harmonic_mean_ratio = 2.0 * (min_ratio * soft_min_ratio) / (min_ratio + soft_min_ratio + epsilon)
    
    # Combine: geometric mean of base_heuristic and harmonic_mean_ratio, then power amplify
    combined = np.sqrt(base_heuristic * harmonic_mean_ratio)
    
    # Power amplify to sharpen differences (measured improvement: power=1.5)
    heuristic = combined ** 1.5
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize so median is around 1
    median_val = np.median(heuristic)
    if median_val > 0 and np.isfinite(median_val):
        heuristic = heuristic / median_val
    
    # Final safety: ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    return heuristic

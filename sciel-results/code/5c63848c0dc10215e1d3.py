import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute heuristic values for ant colony optimization using a competitive
    exclusion / opportunity-cost model for the multiple-constraint knapsack.
    
    The heuristic estimates the marginal utility of each item by considering
    how much prize it provides relative to the competition it faces in each
    constraint dimension. Items that are efficient (high prize per unit weight)
    in their bottleneck constraint and face less competition from other items
    in that constraint are scored higher.
    
    Strategy:
    1. Compute per-constraint efficiency: prize[i] / weight[c][i]
    2. For each item, identify its bottleneck constraint (lowest efficiency)
    3. Estimate competitive pressure: sum of prizes of items with similar or 
       higher efficiency in that constraint
    4. Heuristic = prize / (competitive pressure), smoothed to be positive
    """
    start_time = time.time()
    deadline = 325.0  # safety margin before 330s kill
    
    # Robustly convert to numpy arrays and handle shape
    prizes = np.asarray(prizes)
    weights = np.asarray(weights)
    
    # Expected shape is (5, 100)
    # If 1D, try to reshape if size matches 500
    if prizes.ndim == 1:
        if prizes.size == 500:
            prizes = prizes.reshape(5, 100)
        else:
            # Fallback if shape is unexpected
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
    # Shape: (n_constraints, n_items)
    ratios = prizes / safe_weights
    
    # Clip very small ratios to avoid log issues
    ratios = np.clip(ratios, epsilon, None)
    
    # For each item, find the bottleneck constraint (minimum efficiency)
    # This is the constraint that limits the item the most
    min_ratio = np.min(ratios, axis=0)  # shape: (n_items,)
    
    # Also compute a smooth version using soft-min via log-sum-exp
    # soft_min(x) = -log(sum(exp(-x))) + max(x)  (numerically stable)
    neg_ratios = -np.log(ratios)  # shape: (n_constraints, n_items)
    max_neg = np.max(neg_ratios, axis=0, keepdims=True)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_ratios - max_neg), axis=0))
    soft_min_log_ratio = -log_sum_exp  # shape: (n_items,)
    soft_min_ratio = np.exp(soft_min_log_ratio)
    
    # Competitive exclusion score:
    # For each item, the "competition" in its bottleneck constraint is the
    # total prize of other items that are at least as efficient in that constraint.
    # This is expensive to compute exactly, so we approximate.
    
    # Approximation: for each constraint c, sort items by efficiency ratio.
    # An item's competitive pressure is the sum of prizes of items with
    # ratio >= its own ratio in that constraint (including itself).
    # We use the bottleneck constraint for each item.
    
    # Build the competitive pressure array
    competitive_pressure = np.zeros(n_items)
    
    for c in range(n_constraints):
        # Sort items by ratio in constraint c (descending)
        order = np.argsort(-ratios[c])
        sorted_prizes = prizes[c, order]
        # Cumulative sum from the end (items with lower or equal ratio)
        # Actually, we want sum of prizes of items with ratio >= current item's ratio
        # So cumulative sum from the top
        cumsum_prizes = np.cumsum(sorted_prizes)
        
        # Map back to original indices
        pressure_in_c = np.empty(n_items)
        pressure_in_c[order] = cumsum_prizes
        
        # For items whose bottleneck is constraint c, use this pressure
        bottleneck_mask = np.argmin(ratios, axis=0) == c
        competitive_pressure[bottleneck_mask] = np.where(
            bottleneck_mask, pressure_in_c, 0
        )[bottleneck_mask]
    
    # Ensure competitive pressure is at least the item's own prize (avoid zero division)
    competitive_pressure = np.maximum(competitive_pressure, prizes.sum(axis=0) * 0.01 + epsilon)
    
    # Heuristic: prize / competitive pressure, combined with soft-min ratio
    # This gives high score to items with high prize but low competition
    base_heuristic = prizes.sum(axis=0) / competitive_pressure
    
    # Combine with the soft-min efficiency ratio (which captures balance across constraints)
    # Geometric mean of the two signals
    heuristic = np.sqrt(base_heuristic * soft_min_ratio)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize to a reasonable range to avoid extreme values
    # Scale so that the median is around 1
    median_val = np.median(heuristic)
    if median_val > 0 and np.isfinite(median_val):
        heuristic = heuristic / median_val
    
    # Final safety: ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    return heuristic

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _find_lambda(remaining_capacity, weights, values, max_iters=50):
    """
    Binary search for lambda such that the total weight of items with
    v_i/lambda > w_i (i.e., v_i - lambda*w_i > 0) approximately matches
    remaining_capacity.
    
    Returns lambda value.
    """
    n = len(weights)
    if n == 0:
        return 0.0
    
    # Initial bounds
    # Lower bound: 0 (all items selected)
    # Upper bound: max density + small epsilon
    densities = values / weights
    lambda_lo = 0.0
    lambda_hi = float(np.max(densities)) * 1.5 + 1e-6
    
    # Binary search
    for _ in range(max_iters):
        lambda_mid = (lambda_lo + lambda_hi) / 2.0
        
        # Compute which items are "worth taking" at this lambda
        # Item is worth taking if v_i - lambda * w_i > 0
        scores = values - lambda_mid * weights
        selected_mask = scores > 0
        
        total_weight = float(np.sum(weights[selected_mask]))
        
        if total_weight > remaining_capacity:
            # Too much weight, need higher lambda
            lambda_lo = lambda_mid
        else:
            # Not enough weight (or exactly right), need lower lambda
            lambda_hi = lambda_mid
        
        # Check convergence
        if lambda_hi - lambda_lo < 1e-6:
            break
    
    return (lambda_lo + lambda_hi) / 2.0


def _greedy_fill_vectorized(remaining_capacity, weights, values):
    """
    Fast greedy fill by value density using numpy operations.
    Returns total value obtained.
    """
    if len(weights) == 0 or remaining_capacity <= 0:
        return 0.0
    
    densities = values / weights
    order = np.argsort(-densities)
    
    total_value = 0.0
    cap = remaining_capacity
    
    # Vectorized-ish approach: iterate in sorted order
    for idx in order:
        w = weights[idx]
        v = values[idx]
        if w <= cap:
            cap -= w
            total_value += v
        else:
            # Once we hit an item that doesn't fit, lighter items might still fit
            # but we need to continue checking
            pass
    
    return total_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack using Lagrangian relaxation.
    
    Strategy:
      1. Filter to feasible items (weight <= remaining_capacity).
      2. If none feasible, return -1.
      3. Use binary search to find lambda that matches remaining capacity.
      4. Score each feasible item by v_i - lambda * w_i.
      5. Pick the item with the highest score.
      6. Time guard with fallback to density-based selection.
    """
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Global deadline guard
    if not hasattr(select_next_item, '_start_time'):
        select_next_item._start_time = time.time()
    
    elapsed = time.time() - select_next_item._start_time
    if elapsed > 140:  # Safety margin before 150s kill
        return -1
    
    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    call_start = time.time()
    time_budget = 0.08  # 80ms per call
    
    # Get feasible weights and values
    f_weights = remaining_weights[feasible_indices]
    f_values = remaining_values[feasible_indices]
    
    # Find lambda via binary search
    lambda_val = _find_lambda(remaining_capacity, f_weights, f_values, max_iters=40)
    
    # Check time
    if time.time() - call_start > time_budget:
        # Fallback: pick highest density
        f_densities = f_values / f_weights
        return int(feasible_indices[np.argmax(f_densities)])
    
    # Score items by Lagrangian: v_i - lambda * w_i
    scores = f_values - lambda_val * f_weights
    
    # Only consider items with positive score (worth taking)
    positive_mask = scores > 0
    if not np.any(positive_mask):
        # No item has positive Lagrangian score; fallback to density
        f_densities = f_values / f_weights
        return int(feasible_indices[np.argmax(f_densities)])
    
    # Among positive-score items, pick the one with highest score
    # But also ensure it's feasible (already filtered)
    positive_indices = np.where(positive_mask)[0]
    best_pos = positive_indices[np.argmax(scores[positive_indices])]
    
    return int(feasible_indices[best_pos])


# EVOLVE-BLOCK-END

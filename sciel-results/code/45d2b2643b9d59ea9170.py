"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: evaluator kills after 150s. We keep a safety margin.
_START_TIME = time.time()
_DEADLINE_SECONDS = 140.0  # 10s safety margin before the 150s kill

def _elapsed_ok():
    return (time.time() - _START_TIME) < _DEADLINE_SECONDS

def _fractional_knapsack_bound(capacity, weights, values, exclude_idx=-1):
    """
    Compute the fractional knapsack bound for the given state.
    Returns (bound_value, lambda_estimate) where lambda_estimate is the
    marginal value per unit weight (the Lagrange multiplier).
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, 0.0
    
    feasible_mask = (weights <= capacity)
    if exclude_idx >= 0:
        feasible_mask[exclude_idx] = False
    
    if not np.any(feasible_mask):
        return 0.0, 0.0
    
    feasible_indices = np.where(feasible_mask)[0]
    fw = weights[feasible_indices]
    fv = values[feasible_indices]
    
    # Sort by density (value/weight) descending
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(fw > 0, fv / fw, 0.0)
    order = np.argsort(-dens)
    
    fw_sorted = fw[order]
    fv_sorted = fv[order]
    dens_sorted = dens[order]
    
    sim_cap = capacity
    sim_val = 0.0
    lambda_est = 0.0
    
    for i in range(len(fw_sorted)):
        if fw_sorted[i] <= sim_cap:
            sim_cap -= fw_sorted[i]
            sim_val += fv_sorted[i]
        else:
            # Partial item: this gives us the lambda estimate
            if fw_sorted[i] > 0:
                lambda_est = dens_sorted[i]
            # Add fractional part
            sim_val += sim_cap * dens_sorted[i]
            sim_cap = 0
            break
        if sim_cap <= 0:
            break
    
    # If we filled everything, lambda is the density of the last item added
    if lambda_est == 0.0 and len(fw_sorted) > 0:
        lambda_est = dens_sorted[-1]
    
    return sim_val, lambda_est

def _greedy_fill_fast(capacity, weights, values, order_indices, exclude_idx=-1):
    """
    Fast greedy knapsack fill using the given order of indices.
    Returns the total value achieved.
    """
    sim_cap = capacity
    sim_val = 0.0
    for idx in order_indices:
        if idx == exclude_idx:
            continue
        w = weights[idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += values[idx]
        if sim_cap <= 0:
            break
    return sim_val

def _estimate_future_value_lagrangian(capacity, weights, values, exclude_idx=-1):
    """
    Estimate the future value using the Lagrangian dual approach.
    Uses the lambda estimate from the fractional relaxation to compute
    reduced profits, then greedily selects items with positive reduced profit.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    feasible_mask = (weights <= capacity)
    if exclude_idx >= 0:
        feasible_mask[exclude_idx] = False
    
    if not np.any(feasible_mask):
        return 0.0
    
    feasible_indices = np.where(feasible_mask)[0]
    fw = weights[feasible_indices]
    fv = values[feasible_indices]
    
    # Get lambda estimate from fractional relaxation
    _, lambda_est = _fractional_knapsack_bound(capacity, weights, values, exclude_idx)
    
    if lambda_est <= 0:
        # Fallback to density greedy
        with np.errstate(divide='ignore', invalid='ignore'):
            dens = np.where(fw > 0, fv / fw, 0.0)
        order = feasible_indices[np.argsort(-dens)]
        return _greedy_fill_fast(capacity, weights, values, order, exclude_idx)
    
    # Compute reduced profits: v_i - lambda * w_i
    reduced_profits = fv - lambda_est * fw
    
    # Select items with positive reduced profit, sorted by density
    positive_mask = reduced_profits > 0
    if not np.any(positive_mask):
        return 0.0
    
    positive_indices = feasible_indices[positive_mask]
    pos_fw = fw[positive_mask]
    pos_fv = fv[positive_mask]
    pos_rp = reduced_profits[positive_mask]
    
    # Sort by density (value/weight) descending
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(pos_fw > 0, pos_fv / pos_fw, 0.0)
    order = positive_indices[np.argsort(-dens)]
    
    # Greedily fill with these items
    sim_cap = capacity
    sim_val = 0.0
    for idx in order:
        w = weights[idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += values[idx]
        if sim_cap <= 0:
            break
    
    return sim_val

def _estimate_future_value_hybrid(capacity, weights, values, exclude_idx=-1):
    """
    Hybrid estimate: combines Lagrangian-based greedy with traditional
    density/value/weight greedy strategies, taking the best.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    feasible_mask = (weights <= capacity)
    if exclude_idx >= 0:
        feasible_mask[exclude_idx] = False
    
    if not np.any(feasible_mask):
        return 0.0
    
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return 0.0
    
    fw = weights[feasible_indices]
    fv = values[feasible_indices]
    
    # Strategy 1: Lagrangian-based
    val_lag = _estimate_future_value_lagrangian(capacity, weights, values, exclude_idx)
    
    # Strategy 2: density order
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(fw > 0, fv / fw, 0.0)
    order_density = feasible_indices[np.argsort(-dens)]
    val_density = _greedy_fill_fast(capacity, weights, values, order_density, exclude_idx)
    
    # Strategy 3: value order
    order_value = feasible_indices[np.argsort(-fv)]
    val_value = _greedy_fill_fast(capacity, weights, values, order_value, exclude_idx)
    
    # Strategy 4: weight order (lightest first)
    order_weight = feasible_indices[np.argsort(fw)]
    val_weight = _greedy_fill_fast(capacity, weights, values, order_weight, exclude_idx)
    
    return max(val_lag, val_density, val_value, val_weight)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Safety check: if we've been running too long, fall back to simple greedy
    if not _elapsed_ok():
        # Fallback: simple density greedy
        if len(remaining_weights) == 0 or remaining_capacity <= 0:
            return -1
        weights = np.asarray(remaining_weights, dtype=float)
        values = np.asarray(remaining_values, dtype=float)
        feasible = np.where(weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        # Pick best density among feasible
        with np.errstate(divide='ignore', invalid='ignore'):
            dens = np.where(weights[feasible] > 0, values[feasible] / weights[feasible], 0.0)
        best_local = np.argmax(dens)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Filter to feasible items
    feasible_mask = weights <= c
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Get the Lagrange multiplier estimate from the current state
    _, lambda_est = _fractional_knapsack_bound(c, weights, values)
    
    if lambda_est <= 0:
        # Fallback: use density
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = np.where(weights > 0, values / weights, 0.0)
        # Pick best density among feasible
        feas_dens = densities[feasible_indices]
        best_local = np.argmax(feas_dens)
        return int(feasible_indices[best_local])
    
    best_total_val = -1.0
    best_idx = -1
    
    # For each feasible item, compute total estimated value using Lagrangian-based
    # future value estimation
    for idx in feasible_indices:
        if not _elapsed_ok():
            break
            
        w_i = weights[idx]
        v_i = values[idx]
        remaining_cap = c - w_i
        
        if remaining_cap < 0:
            continue
        
        # Estimate future value using hybrid Lagrangian approach
        future_val = _estimate_future_value_hybrid(remaining_cap, weights, values, idx)
        
        # Total estimated value if we pick this item now
        total_est = v_i + future_val
        
        if total_est > best_total_val:
            best_total_val = total_est
            best_idx = int(idx)
    
    if best_idx == -1:
        # Fallback: pick best reduced profit among feasible
        reduced_profits = values - lambda_est * weights
        feas_rp = reduced_profits[feasible_indices]
        best_local = np.argmax(feas_rp)
        return int(feasible_indices[best_local])
        
    return best_idx
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: evaluator kills after 150s. We keep a safety margin.
_START_TIME = time.time()
_DEADLINE_SECONDS = 140.0  # 10s safety margin before the 150s kill

def _elapsed_ok():
    return (time.time() - _START_TIME) < _DEADLINE_SECONDS

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

def _fractional_knapsack_bound(capacity, weights, values, exclude_idx=-1):
    """
    Compute the fractional knapsack upper bound (LP relaxation).
    This provides a tighter, smoother estimate than pure greedy heuristics.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    feasible_mask = (weights <= capacity) & (weights > 0)
    if exclude_idx >= 0:
        feasible_mask[exclude_idx] = False
    
    if not np.any(feasible_mask):
        return 0.0
    
    indices = np.where(feasible_mask)[0]
    fw = weights[indices]
    fv = values[indices]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(fw > 0, fv / fw, 0.0)
    
    order = indices[np.argsort(-dens)]
    
    sim_cap = capacity
    total_val = 0.0
    for idx in order:
        w = weights[idx]
        v = values[idx]
        if w <= sim_cap:
            sim_cap -= w
            total_val += v
        else:
            total_val += v * (sim_cap / w)
            break
    
    return total_val

def _estimate_future_value(capacity, weights, values, exclude_idx=-1):
    """
    Estimate the future value achievable after excluding one item, using
    multiple greedy strategies plus a fractional knapsack bound.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Filter to feasible items
    feasible_mask = (weights <= capacity)
    if exclude_idx >= 0:
        feasible_mask[exclude_idx] = False
    
    if not np.any(feasible_mask):
        return 0.0
    
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return 0.0
    
    # Get weights and values for feasible items only
    fw = weights[feasible_indices]
    fv = values[feasible_indices]
    
    # Strategy 1: density order
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(fw > 0, fv / fw, 0.0)
    order_density = feasible_indices[np.argsort(-dens)]
    
    # Strategy 2: value order
    order_value = feasible_indices[np.argsort(-fv)]
    
    # Strategy 3: weight order (lightest first)
    order_weight = feasible_indices[np.argsort(fw)]
    
    # Strategy 4: value-per-weight-squared (penalizes heavy items more)
    with np.errstate(divide='ignore', invalid='ignore'):
        dens2 = np.where(fw > 0, fv / (fw * fw + 1e-10), 0.0)
    order_weight_sq = feasible_indices[np.argsort(-dens2)]
    
    # Simulate all four and take the best
    val_1 = _greedy_fill_fast(capacity, weights, values, order_density, exclude_idx)
    val_2 = _greedy_fill_fast(capacity, weights, values, order_value, exclude_idx)
    val_3 = _greedy_fill_fast(capacity, weights, values, order_weight, exclude_idx)
    val_4 = _greedy_fill_fast(capacity, weights, values, order_weight_sq, exclude_idx)
    
    greedy_best = max(val_1, val_2, val_3, val_4)
    
    # Also compute fractional knapsack bound (upper bound)
    frac_bound = _fractional_knapsack_bound(capacity, weights, values, exclude_idx)
    
    # Use a blend: the fractional bound is an upper bound, greedy is achievable.
    # A weighted average gives a smoother surrogate that's closer to true 0/1 value.
    # 0.7 * greedy + 0.3 * fractional tends to be a good estimate.
    estimate = 0.7 * greedy_best + 0.3 * frac_bound
    
    return estimate

def _two_step_lookahead(capacity, weights, values, first_idx):
    """
    Evaluate a 2-step lookahead: pick first_idx, then pick the best second item
    based on 1-step lookahead from the remaining state.
    Returns the total estimated value.
    """
    n = len(weights)
    w1 = weights[first_idx]
    v1 = values[first_idx]
    cap_after = capacity - w1
    
    if cap_after < 0:
        return v1  # just the first item's value
    
    # For the second step, evaluate each remaining feasible item
    feasible_mask = (weights <= cap_after) & (np.arange(n) != first_idx)
    if not np.any(feasible_mask):
        return v1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return v1
    
    # Quick estimate: use fractional knapsack bound for the remainder
    frac_bound = _fractional_knapsack_bound(cap_after, weights, values, first_idx)
    
    # Also do a quick 1-step greedy to get an achievable lower bound
    fw = weights[feasible_indices]
    fv = values[feasible_indices]
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(fw > 0, fv / fw, 0.0)
    order_density = feasible_indices[np.argsort(-dens)]
    greedy_val = _greedy_fill_fast(cap_after, weights, values, order_density, first_idx)
    
    # Blend for smoother estimate
    second_step_est = 0.6 * greedy_val + 0.4 * frac_bound
    
    return v1 + second_step_est

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
    
    # Pre-compute sorting orders for multiple greedy strategies
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0.0)
    
    order_density = np.argsort(-densities)
    order_value = np.argsort(-values)
    order_weight = np.argsort(weights)
    
    best_total_val = -1.0
    best_idx = -1
    
    # Phase 1: 1-step lookahead for all feasible items
    scores = {}
    for idx in feasible_indices:
        if not _elapsed_ok():
            break
            
        w_i = weights[idx]
        v_i = values[idx]
        remaining_cap = c - w_i
        
        if remaining_cap < 0:
            continue
        
        future_val = _estimate_future_value(remaining_cap, weights, values, idx)
        total_est = v_i + future_val
        
        scores[idx] = total_est
        if total_est > best_total_val:
            best_total_val = total_est
            best_idx = int(idx)
    
    if best_idx == -1:
        return -1
    
    # Phase 2: 2-step lookahead polish on top-5 candidates
    if _elapsed_ok() and len(feasible_indices) > 5:
        # Sort by 1-step score, take top 5
        sorted_candidates = sorted(scores.items(), key=lambda x: -x[1])[:5]
        
        for idx, score_1step in sorted_candidates:
            if not _elapsed_ok():
                break
            
            score_2step = _two_step_lookahead(c, weights, values, idx)
            
            # Blend 1-step and 2-step scores (2-step is more accurate but noisier)
            blended = 0.4 * score_1step + 0.6 * score_2step
            
            if blended > best_total_val:
                best_total_val = blended
                best_idx = int(idx)
    
    return best_idx
# EVOLVE-BLOCK-END

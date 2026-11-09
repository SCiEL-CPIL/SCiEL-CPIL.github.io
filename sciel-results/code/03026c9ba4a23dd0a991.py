# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety
_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 2.0

def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)

def _fractional_value(capacity, weights, values, order_indices, exclude_idx):
    """
    Calculates the value of a fractional knapsack solution
    for the remaining capacity, using the given order, excluding one item.
    """
    sim_cap = capacity
    sim_val = 0.0
    if sim_cap <= 0:
        return 0.0
    for idx in order_indices:
        if idx == exclude_idx:
            continue
        w = weights[idx]
        v = values[idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += v
        else:
            # Take fraction
            sim_val += v * (sim_cap / w)
            sim_cap = 0
            break
    return sim_val

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
        dens = values[feasible] / weights[feasible]
        best_local = np.argmax(dens)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    # Filter to feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Pre-compute sorting order by Density (Value / Weight)
    # This is the standard optimal order for Fractional Knapsack
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0.0)
    order_density = np.argsort(-densities)
    
    best_total_val = -1.0
    best_idx = -1
    
    # For each feasible item, estimate the total value if we pick it now
    # and then fill the rest with the Fractional Knapsack optimal (upper bound)
    for idx in feasible_indices:
        if not _elapsed_ok():
            break
            
        w_i = weights[idx]
        v_i = values[idx]
        remaining_cap = remaining_capacity - w_i
        
        # Simulate Fractional Fill for the rest
        # This acts as a smooth surrogate for the potential value
        val_sim = _fractional_value(remaining_cap, weights, values, order_density, idx)
        
        total_val = v_i + val_sim
        
        if total_val > best_total_val:
            best_total_val = total_val
            best_idx = idx
            
    if best_idx == -1:
        return -1
        
    return int(best_idx)
# EVOLVE-BLOCK-END

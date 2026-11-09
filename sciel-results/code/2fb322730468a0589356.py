import numpy as np
import time

def _greedy_fill(capacity, weights, values, exclude_idx, order):
    """Greedy fill: iterate items in given order, pick if fits."""
    total_val = 0.0
    curr_cap = capacity
    for j in order:
        if j == exclude_idx:
            continue
        if weights[j] <= curr_cap:
            curr_cap -= weights[j]
            total_val += values[j]
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Direct heuristic for 0/1 knapsack: pick the item that, when combined with a
    greedy fill of the remaining capacity, yields the highest total value.
    Uses multiple fill strategies (density-order, value-order) to improve lookahead.
    """
    start_time = time.time()
    deadline = 145.0  # 5-second safety margin before 150s kill
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    
    n = len(weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Quick check: if no item fits, stop
    if np.all(weights > remaining_capacity):
        return -1
    
    # Filter to only feasible items (weight <= remaining_capacity)
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
    
    # Compute densities (value / weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(f_weights > 0, f_values / f_weights, np.inf)
    
    # Sort by density descending; tie-break by higher value, then lower weight
    density_order = np.lexsort((f_weights, -f_values, -densities))
    
    # Sort by value descending; tie-break by lower weight (better density)
    value_order = np.lexsort((f_weights, -f_values))
    
    # Combined score: blend density and value for a third ordering
    # Normalize values to [0,1] range for blending
    min_val, max_val = f_values.min(), f_values.max()
    val_range = max_val - min_val
    if val_range > 0:
        norm_values = (f_values - min_val) / val_range
    else:
        norm_values = np.ones(f_n)
    
    # Blend: 0.7 * normalized_density + 0.3 * normalized_value
    min_dens, max_dens = densities.min(), densities.max()
    dens_range = max_dens - min_dens
    if dens_range > 0:
        norm_dens = (densities - min_dens) / dens_range
    else:
        norm_dens = np.ones(f_n)
    
    blend_score = 0.7 * norm_dens + 0.3 * norm_values
    blend_order = np.lexsort((f_weights, -f_values, -blend_score))
    
    # Precompute sorted arrays for density order (primary candidate selection)
    sorted_f_indices = feasible_indices[density_order]
    sorted_weights = f_weights[density_order]
    sorted_values = f_values[density_order]
    
    # Determine K based on time budget and problem size
    # Estimate cost per candidate: O(f_n) for each fill strategy
    # With 3 fill strategies, cost per candidate ~ 3 * f_n
    # Target: use ~80% of available time for lookahead
    elapsed = time.time() - start_time
    remaining_time = deadline - 5.0 - elapsed
    if remaining_time <= 0:
        K = min(f_n, 5)
    else:
        # Estimate: each candidate takes ~3*f_n operations
        # Assume ~1e8 simple ops/sec in Python+NumPy mix
        max_candidates = int(remaining_time * 1e8 / (3 * f_n))
        K = min(f_n, max(10, min(max_candidates, 100)))
    
    best_total_value = -1
    best_global_idx = -1
    
    for k in range(K):
        # Check deadline
        if time.time() - start_time > deadline - 5.0:
            break
        
        i = k
        if sorted_weights[i] > remaining_capacity:
            continue
        
        # Simulate picking item i
        curr_cap = remaining_capacity - sorted_weights[i]
        base_val = sorted_values[i]
        
        # Try three fill strategies and take the best
        fill1 = _greedy_fill(curr_cap, f_weights, f_values, i, density_order)
        total1 = base_val + fill1
        
        fill2 = _greedy_fill(curr_cap, f_weights, f_values, i, value_order)
        total2 = base_val + fill2
        
        fill3 = _greedy_fill(curr_cap, f_weights, f_values, i, blend_order)
        total3 = base_val + fill3
        
        best_fill_total = max(total1, total2, total3)
        
        if best_fill_total > best_total_value:
            best_total_value = best_fill_total
            best_global_idx = int(sorted_f_indices[i])
    
    if best_global_idx == -1:
        return -1
    
    return best_global_idx

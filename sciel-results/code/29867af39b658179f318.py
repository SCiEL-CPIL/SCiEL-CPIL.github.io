"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_DEADLINE = None


def _get_deadline():
    """Get or set the global deadline (150s wall-clock with safety margin)."""
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 145.0
    return _GLOBAL_DEADLINE


def _find_lagrangian_boundary(weights, values, capacity):
    """
    Find the Lagrangian multiplier and identify boundary items.
    Returns (lambda_val, boundary_indices, sorted_order)
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, [], []
    
    densities = values / weights
    order = np.argsort(-densities, kind='stable')
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    
    # Find where cumulative weight exceeds capacity
    idx = np.searchsorted(cum_w, capacity, side='right')
    
    if idx == 0:
        return densities[order[0]] + 1.0, [order[0]], order
    if idx == n:
        return 0.0, [], order
    
    # The boundary item is at position idx (the one that would be fractional)
    lambda_val = densities[order[idx]]
    
    # Boundary region: items near the cutoff
    # Take items around the boundary
    start = max(0, idx - 5)
    end = min(n, idx + 6)
    boundary_indices = order[start:end].tolist()
    
    return lambda_val, boundary_indices, order


def _dp_fill(capacity, weights, values, excluded):
    """
    Exact DP fill for small capacities.
    capacity: integer capacity (scaled)
    weights, values: arrays
    excluded: set of indices to exclude
    Returns best value achievable.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Scale: assume weights are already in reasonable range
    # For capacity 12.5, scale by 10 to get integer capacity 125
    cap_int = int(capacity)
    if cap_int <= 0:
        return 0.0
    
    # DP table: dp[c] = max value with capacity c
    dp = np.zeros(cap_int + 1, dtype=np.float64)
    
    for i in range(n):
        if i in excluded:
            continue
        w = int(weights[i])
        v = values[i]
        if w <= 0:
            continue
        if w > cap_int:
            continue
        # Update dp in reverse
        for c in range(cap_int, w - 1, -1):
            new_val = dp[c - w] + v
            if new_val > dp[c]:
                dp[c] = new_val
    
    return float(dp[cap_int])


def _greedy_fill_by_density(remaining_capacity, remaining_weights, remaining_values, order, excluded):
    """Greedy fill by density order, excluding specified items."""
    cap = remaining_capacity
    total_value = 0.0
    
    for idx in order:
        if idx in excluded:
            continue
        w = remaining_weights[idx]
        if w <= cap:
            total_value += remaining_values[idx]
            cap -= w
        if cap <= 0:
            break
    
    return total_value


def _one_step_lookahead(remaining_capacity, remaining_weights, remaining_values, order, boundary_indices, deadline):
    """
    1-step lookahead: try each candidate, then fill greedily.
    Prioritize boundary items and high-density items.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    n_feasible = len(feasible_indices)
    
    if n_feasible == 1:
        return int(feasible_indices[0])
    
    # Build candidate set: boundary items + top density feasible items
    boundary_set = set(boundary_indices)
    boundary_feasible = [i for i in boundary_indices if feasible_mask[i]]
    
    # Top density feasible items
    densities = remaining_values[feasible_indices] / remaining_weights[feasible_indices]
    top_density_count = min(15, n_feasible)
    top_density_local = np.argsort(-densities)[:top_density_count]
    top_density_global = feasible_indices[top_density_local]
    
    # Combine candidates
    candidates = list(set(boundary_feasible + list(top_density_global)))
    
    # Limit candidates
    max_candidates = 40
    if len(candidates) > max_candidates:
        # Prioritize boundary items
        boundary_candidates = [c for c in candidates if c in boundary_set]
        other_candidates = [c for c in candidates if c not in boundary_set]
        candidates = boundary_candidates[:20] + other_candidates[:max_candidates - len(boundary_candidates[:20])]
    
    best_score = -1.0
    best_idx = -1
    
    for cand_idx in candidates:
        if time.time() > deadline:
            break
        
        cand_w = remaining_weights[cand_idx]
        cand_v = remaining_values[cand_idx]
        cap_after = remaining_capacity - cand_w
        
        if cap_after < 0:
            continue
        
        # Try DP fill if capacity is small enough
        if cap_after <= 15.0:
            total_v = cand_v + _dp_fill(cap_after, remaining_weights, remaining_values, {cand_idx})
        else:
            total_v = cand_v + _greedy_fill_by_density(cap_after, remaining_weights, remaining_values, order, {cand_idx})
        
        if total_v > best_score:
            best_score = total_v
            best_idx = int(cand_idx)
    
    if best_idx != -1:
        return best_idx
    
    # Fallback: best density feasible
    return int(feasible_indices[np.argmax(densities)])


def _two_step_lookahead(remaining_capacity, remaining_weights, remaining_values, order, boundary_indices, deadline, top_k=4):
    """
    2-step lookahead: try top-k first picks, then for each, try best second pick.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    n_feasible = len(feasible_indices)
    
    if n_feasible == 1:
        return int(feasible_indices[0])
    
    densities = remaining_values / remaining_weights
    feasible_densities = densities[feasible_indices]
    
    # Get top candidates by density and boundary
    boundary_set = set(boundary_indices)
    boundary_feasible = [i for i in boundary_indices if feasible_mask[i]]
    
    top_density_count = min(10, n_feasible)
    top_density_local = np.argsort(-feasible_densities)[:top_density_count]
    top_density_global = feasible_indices[top_density_local]
    
    first_candidates = list(set(boundary_feasible + list(top_density_global)))
    first_candidates = first_candidates[:top_k]
    
    best_total = -1.0
    best_idx = -1
    
    for cand1 in first_candidates:
        if time.time() > deadline:
            break
        
        cand1_w = remaining_weights[cand1]
        cand1_v = remaining_values[cand1]
        cap_after1 = remaining_capacity - cand1_w
        
        if cap_after1 < 0:
            continue
        
        # For second pick, try boundary items and top density
        second_candidates = list(set(boundary_feasible + list(top_density_global)))
        second_candidates = [c for c in second_candidates if c != cand1 and remaining_weights[c] <= cap_after1]
        second_candidates = second_candidates[:15]
        
        for cand2 in second_candidates:
            if time.time() > deadline:
                break
            
            cand2_w = remaining_weights[cand2]
            cand2_v = remaining_values[cand2]
            cap_after2 = cap_after1 - cand2_w
            
            if cap_after2 < 0:
                continue
            
            total_v = cand1_v + cand2_v
            
            # Fill rest
            if cap_after2 <= 15.0:
                total_v += _dp_fill(cap_after2, remaining_weights, remaining_values, {cand1, cand2})
            else:
                total_v += _greedy_fill_by_density(cap_after2, remaining_weights, remaining_values, order, {cand1, cand2})
            
            if total_v > best_total:
                best_total = total_v
                best_idx = int(cand1)
    
    if best_idx != -1:
        return best_idx
    
    # Fallback: best density
    return int(feasible_indices[np.argmax(feasible_densities)])


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Returns an index into the remaining (feasible) arrays, or -1 to stop.
    
    Strategy:
      1. Compute Lagrangian boundary to identify critical items.
      2. Use 2-step lookahead for small problem sizes, 1-step for larger.
      3. Use DP fill for small remaining capacities.
    """
    if len(remaining_weights) == 0 or remaining_capacity <= 0:
        return -1
    
    deadline = _get_deadline()
    
    w_arr = np.asarray(remaining_weights, dtype=np.float64)
    v_arr = np.asarray(remaining_values, dtype=np.float64)
    
    # Feasibility check
    feasible_mask = w_arr <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    n_feasible = int(np.sum(feasible_mask))
    
    if n_feasible == 1:
        return int(np.where(feasible_mask)[0][0])
    
    # Check time budget
    if time.time() > deadline - 5.0:
        # Not enough time, use pure density
        densities = v_arr[feasible_mask] / w_arr[feasible_mask]
        best_local = int(np.argmax(densities))
        return int(np.where(feasible_mask)[0][best_local])
    
    # Compute Lagrangian boundary
    w_feas = w_arr[feasible_mask]
    v_feas = v_arr[feasible_mask]
    
    # If all feasible items fit, pick highest density
    if np.sum(w_feas) <= remaining_capacity:
        densities = v_arr[feasible_mask] / w_arr[feasible_mask]
        best_local = int(np.argmax(densities))
        return int(np.where(feasible_mask)[0][best_local])
    
    lambda_val, boundary_indices, order = _find_lagrangian_boundary(w_feas, v_feas, remaining_capacity)
    
    # Convert boundary_indices to global indices
    feasible_indices = np.where(feasible_mask)[0]
    global_boundary = []
    for i in boundary_indices:
        if i < len(feasible_indices):
            global_boundary.append(int(feasible_indices[i]))
    
    # Choose lookahead depth based on problem size and time
    if n_feasible <= 25 and time.time() < deadline - 2.0:
        result = _two_step_lookahead(remaining_capacity, w_arr, v_arr, order, global_boundary, deadline, top_k=4)
    else:
        result = _one_step_lookahead(remaining_capacity, w_arr, v_arr, order, global_boundary, deadline)
    
    # Validate result
    if result == -1 or result < 0 or result >= len(w_arr):
        # Fallback
        densities = v_arr[feasible_mask] / w_arr[feasible_mask]
        best_local = int(np.argmax(densities))
        return int(np.where(feasible_mask)[0][best_local])
    
    return result
# EVOLVE-BLOCK-END

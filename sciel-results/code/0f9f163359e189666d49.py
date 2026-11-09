import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill
_CALL_START = [0.0]

def _time_remaining():
    return _DEADLINE - time.time()

def _initial_greedy_solution(capacity, weights, values):
    """Generate an initial solution using density greedy."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return set(), 0.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / np.where(weights > 0, weights, 1.0)
    
    order = np.argsort(-densities)
    
    selected = set()
    remaining_cap = capacity
    total_value = 0.0
    
    for idx in order:
        w = weights[idx]
        if w <= remaining_cap:
            selected.add(idx)
            remaining_cap -= w
            total_value += values[idx]
            
    return selected, total_value

def _greedy_by_value(capacity, weights, values):
    """Greedy by value descending."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return set(), 0.0
    
    order = np.argsort(-values)
    selected = set()
    remaining_cap = capacity
    total_value = 0.0
    
    for idx in order:
        w = weights[idx]
        if w <= remaining_cap:
            selected.add(idx)
            remaining_cap -= w
            total_value += values[idx]
            
    return selected, total_value

def _greedy_by_weight(capacity, weights, values):
    """Greedy by weight ascending (lightest first)."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return set(), 0.0
    
    order = np.argsort(weights)
    selected = set()
    remaining_cap = capacity
    total_value = 0.0
    
    for idx in order:
        w = weights[idx]
        if w <= remaining_cap:
            selected.add(idx)
            remaining_cap -= w
            total_value += values[idx]
            
    return selected, total_value

def _improve_solution(selected, capacity, weights, values, time_limit=1.0):
    """Improve solution using 1-swap and 2-swap hill climbing with numpy."""
    n = len(weights)
    if n == 0:
        return selected, 0.0
    
    selected_list = list(selected)
    unselected_list = [i for i in range(n) if i not in selected]
    
    current_value = sum(values[i] for i in selected_list)
    current_weight = sum(weights[i] for i in selected_list)
    
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_delta = 0
        best_swap = None
        
        slack = capacity - current_weight
        
        # 1-swap: for each unselected j, find best selected i to remove
        # Vectorized: for each j, compute which i's are feasible and find min value
        if len(unselected_list) > 0 and len(selected_list) > 0:
            sel_arr = np.array(selected_list)
            unsel_arr = np.array(unselected_list)
            
            w_sel = weights[sel_arr]
            v_sel = values[sel_arr]
            w_unsel = weights[unsel_arr]
            v_unsel = values[unsel_arr]
            
            # For each unselected j, min_w_to_remove = w_j - slack
            min_w_to_remove = w_unsel - slack
            
            # Feasibility: w_sel[i] >= min_w_to_remove[j]
            # Value improvement: v_unsel[j] > v_sel[i]
            
            # Create matrix: feasible[j, i] = w_sel[i] >= min_w_to_remove[j]
            # delta[j, i] = v_unsel[j] - v_sel[i]
            
            # To save memory, process in chunks or use broadcasting carefully
            # N=100, so this is fine
            feasible = (w_sel[None, :] >= min_w_to_remove[:, None])
            delta_matrix = v_unsel[:, None] - v_sel[None, :]
            
            # Only consider positive deltas where feasible
            delta_matrix[~feasible] = -np.inf
            delta_matrix[delta_matrix <= 0] = -np.inf
            
            if np.any(np.isfinite(delta_matrix)):
                max_delta = np.max(delta_matrix)
                if max_delta > best_delta:
                    best_delta = max_delta
                    j_idx, i_idx = np.unravel_index(np.argmax(delta_matrix), delta_matrix.shape)
                    best_swap = (selected_list[i_idx], unselected_list[j_idx])
        
        if best_swap:
            i_out, j_in = best_swap
            selected_list.remove(i_out)
            unselected_list.remove(j_in)
            selected_list.append(j_in)
            unselected_list.append(i_out)
            current_value += best_delta
            current_weight += weights[j_in] - weights[i_out]
            improved = True
    
    return set(selected_list), current_value

def _solve_subproblem(capacity, weights, values, time_limit=1.0):
    """Solve a subproblem using multiple heuristics and local search."""
    if len(weights) == 0 or capacity <= 0:
        return 0.0
    
    # Try multiple initial heuristics
    candidates = []
    
    sel1, val1 = _initial_greedy_solution(capacity, weights, values)
    candidates.append((sel1, val1))
    
    sel2, val2 = _greedy_by_value(capacity, weights, values)
    candidates.append((sel2, val2))
    
    sel3, val3 = _greedy_by_weight(capacity, weights, values)
    candidates.append((sel3, val3))
    
    # Pick best initial solution
    best_sel = max(candidates, key=lambda x: x[1])[0]
    best_val = max(x[1] for x in candidates)
    
    # Improve with local search
    if time_limit > 0.1:
        final_sel, final_val = _improve_solution(best_sel, capacity, weights, values, time_limit=time_limit)
        return final_val
    
    return best_val

def _upper_bound_fractional(capacity, weights, values):
    """Fractional knapsack upper bound."""
    if len(weights) == 0 or capacity <= 0:
        return 0.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / np.where(weights > 0, weights, 1.0)
    
    order = np.argsort(-densities)
    
    total = 0.0
    cap = capacity
    
    for idx in order:
        w = weights[idx]
        v = values[idx]
        if w <= cap:
            total += v
            cap -= w
        else:
            total += v * (cap / w)
            break
    
    return total

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Emergency: just pick the best density feasible item
        feasible_mask = remaining_weights <= remaining_capacity
        feasible_idx = np.where(feasible_mask)[0]
        if len(feasible_idx) == 0:
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = remaining_values[feasible_idx] / np.where(remaining_weights[feasible_idx] > 0, remaining_weights[feasible_idx], 1.0)
        return int(feasible_idx[np.argmax(densities)])
    
    _CALL_START[0] = time.time()
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_idx = np.where(feasible_mask)[0]
    
    if len(feasible_idx) == 0:
        return -1
    
    w = remaining_weights
    v = remaining_values
    
    # Prune candidates: only consider top K by some heuristic
    # Use density as initial filter
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v[feasible_idx] / np.where(w[feasible_idx] > 0, w[feasible_idx], 1.0)
    
    # Sort by density descending
    sorted_order = np.argsort(-densities)
    sorted_feasible = feasible_idx[sorted_order]
    
    # Limit candidates to top 30 or all if fewer
    max_candidates = min(30, len(sorted_feasible))
    candidates = sorted_feasible[:max_candidates]
    
    best_total = -1.0
    best_idx = -1
    
    # Time budget for this call
    call_time_limit = min(10.0, max(1.0, _time_remaining() * 0.8))
    
    for candidate in candidates:
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Simulate picking candidate
        new_cap = remaining_capacity - w[candidate]
        val_gain = v[candidate]
        
        # Prepare remaining arrays
        mask = np.ones(n, dtype=bool)
        mask[candidate] = False
        new_w = w[mask]
        new_v = v[mask]
        
        # Quick upper bound check: if even fractional bound + val_gain <= best_total, skip
        if new_cap > 0 and len(new_w) > 0:
            ub = _upper_bound_fractional(new_cap, new_w, new_v)
            if val_gain + ub <= best_total:
                continue
        
        # Run local search on subproblem with limited time
        remaining_time = max(0.1, min(1.0, _time_remaining() * 0.3))
        
        if new_cap <= 0 or len(new_w) == 0:
            total = val_gain
        else:
            total = val_gain + _solve_subproblem(new_cap, new_w, new_v, time_limit=remaining_time)
        
        if total > best_total:
            best_total = total
            best_idx = candidate
    
    return int(best_idx)

import numpy as np
import time

_start_time = None
_GLOBAL_DEADLINE = 135.0  # Leave 15s buffer
_bnb_solution = None  # Store B&B solution (set of indices)
_bnb_initialized = False

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_time_up():
    return _get_elapsed() > _GLOBAL_DEADLINE

def _fractional_knapsack_upper_bound(weights, values, capacity):
    """Compute fractional knapsack upper bound."""
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    densities = values / np.where(weights > 0, weights, 1e-10)
    order = np.argsort(-densities)
    total = 0.0
    rem = capacity
    for i in order:
        if rem <= 0:
            break
        if weights[i] <= rem:
            total += values[i]
            rem -= weights[i]
        else:
            total += values[i] * (rem / weights[i])
            rem = 0
    return total

def _solve_bb(weights, values, capacity, time_budget=2.0):
    """
    Branch and Bound for 0/1 knapsack.
    Returns set of indices in the optimal (or near-optimal) solution.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return set()
    
    # Sort by density descending for better B&B
    densities = values / np.where(weights > 0, weights, 1e-10)
    order = np.argsort(-densities)
    sorted_w = weights[order]
    sorted_v = values[order]
    
    # Compute suffix sums for quick upper bound
    suffix_val = np.zeros(n + 1)
    suffix_w = np.zeros(n + 1)
    for i in range(n - 1, -1, -1):
        suffix_val[i] = suffix_val[i + 1] + sorted_v[i]
        suffix_w[i] = suffix_w[i + 1] + sorted_w[i]
    
    best_val = 0.0
    best_solution = [0] * n
    start_time = time.time()
    
    # DFS with B&B
    stack = [(0, 0.0, 0.0, [0] * n)]  # (idx, current_w, current_v, solution)
    
    while stack:
        if time.time() - start_time > time_budget:
            break
        if _is_time_up():
            break
        
        idx, cur_w, cur_v, sol = stack.pop()
        
        if idx >= n:
            if cur_v > best_val:
                best_val = cur_v
                best_solution = sol[:]
            continue
        
        # Upper bound: current + fractional knapsack of remaining
        rem_cap = capacity - cur_w
        if rem_cap <= 0:
            if cur_v > best_val:
                best_val = cur_v
                best_solution = sol[:]
            continue
        
        # Quick upper bound using suffix
        if cur_v + suffix_val[idx] <= best_val:
            continue
        
        # More precise upper bound with fractional knapsack
        # Only compute if close to best
        if cur_v + suffix_val[idx] > best_val * 0.95:
            ub = _fractional_knapsack_upper_bound(sorted_w[idx:], sorted_v[idx:], rem_cap)
            if cur_v + ub <= best_val:
                continue
        
        # Branch: take item idx
        sol2 = sol[:]
        sol2[idx] = 1
        if sorted_w[idx] <= rem_cap:
            stack.append((idx + 1, cur_w + sorted_w[idx], cur_v + sorted_v[idx], sol2))
        
        # Branch: don't take item idx
        stack.append((idx + 1, cur_w, cur_v, sol))
    
    # Map back to original indices
    result = set()
    for i in range(n):
        if best_solution[i] == 1:
            result.add(int(order[i]))
    
    return result

def _greedy_completion(weights, values, capacity, excluded_idx):
    """
    Greedy completion: given a set of items (with one excluded), 
    greedily pick items by density until capacity is full.
    Returns the total value achieved.
    """
    if capacity <= 0:
        return 0.0
    
    mask = np.ones(len(weights), dtype=bool)
    mask[excluded_idx] = False
    w = weights[mask]
    v = values[mask]
    
    if len(w) == 0:
        return 0.0
    
    densities = v / np.where(w > 0, w, 1e-10)
    order = np.argsort(-densities)
    sorted_w = w[order]
    sorted_v = v[order]
    
    total_val = 0.0
    remaining_cap = capacity
    
    for i in range(len(sorted_w)):
        if sorted_w[i] <= remaining_cap:
            total_val += sorted_v[i]
            remaining_cap -= sorted_w[i]
        else:
            break
    
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _bnb_solution, _bnb_initialized
    
    if _is_time_up():
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # On the first call (or when we have many items), try B&B
    if not _bnb_initialized and n >= 50:
        _bnb_initialized = True
        # Time budget for B&B: use up to 3 seconds
        time_left = _GLOBAL_DEADLINE - _get_elapsed()
        if time_left > 5.0:
            bb_time = min(3.0, time_left - 5.0)
            _bnb_solution = _solve_bb(weights, values, remaining_capacity, time_budget=bb_time)
        else:
            _bnb_solution = None
    
    # If we have a B&B solution, follow it
    if _bnb_solution is not None and len(_bnb_solution) > 0:
        # Find items in the B&B solution that are still available
        for idx in _bnb_solution:
            if 0 <= idx < n:
                if weights[idx] <= remaining_capacity and weights[idx] > 0:
                    return int(idx)
        # B&B solution items not available, fall through to heuristic
    
    # Handle zero weight positive value items first
    zero_weight_indices = np.where((weights == 0) & (values > 0))[0]
    if len(zero_weight_indices) > 0:
        return int(zero_weight_indices[0])
    
    # Filter feasible items (weight > 0 and weight <= capacity)
    feasible_mask = (weights <= remaining_capacity) & (weights > 0)
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
    
    # Lookahead strategy: evaluate top-k density candidates
    C = remaining_capacity
    
    densities = f_values / f_weights
    density_order = np.argsort(-densities)
    
    # Evaluate top-k candidates
    k = min(15, f_n)
    
    best_candidate = -1
    best_total_value = -1.0
    
    for rank in range(k):
        idx = density_order[rank]
        
        new_capacity = C - f_weights[idx]
        if new_capacity < 0:
            continue
        
        # Use fractional knapsack upper bound as a proxy for achievable value
        # This is faster than greedy and more accurate
        remaining_w = np.delete(f_weights, idx)
        remaining_v = np.delete(f_values, idx)
        if len(remaining_w) > 0 and new_capacity > 0:
            completion_value = _fractional_knapsack_upper_bound(remaining_w, remaining_v, new_capacity)
        else:
            completion_value = 0.0
        
        total_value = f_values[idx] + completion_value
        
        if total_value > best_total_value:
            best_total_value = total_value
            best_candidate = idx
    
    if best_candidate == -1:
        best_candidate = density_order[0]
    
    orig_idx = feasible_indices[best_candidate]
    return int(orig_idx)

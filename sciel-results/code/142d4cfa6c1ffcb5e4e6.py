"""Improved 0/1 knapsack heuristic with fractional surrogate lookahead and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 145.0  # seconds; safety margin before 150s kill
_START = None


def _fractional_knapsack_value(weights, values, capacity, exclude_idx=None):
    """
    Compute the optimal fractional knapsack value (upper bound for 0/1).
    Optionally exclude one item index.
    """
    if capacity <= 0:
        return 0.0
    n = len(weights)
    if n == 0:
        return 0.0
    
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        w = weights[mask]
        v = values[mask]
    else:
        w = weights
        v = values
    
    if len(w) == 0:
        return 0.0
    
    # Filter out items with zero or negative weight
    valid = w > 0
    if not np.any(valid):
        return 0.0
    w = w[valid]
    v = v[valid]
    
    if len(w) == 0:
        return 0.0
    
    density = v / w
    order = np.argsort(density)[::-1]
    w_sorted = w[order]
    v_sorted = v[order]
    
    total_value = 0.0
    remaining_cap = capacity
    
    for i in range(len(w_sorted)):
        if remaining_cap <= 0:
            break
        if w_sorted[i] <= remaining_cap:
            total_value += v_sorted[i]
            remaining_cap -= w_sorted[i]
        else:
            total_value += v_sorted[i] * (remaining_cap / w_sorted[i])
            remaining_cap = 0
            break
    
    return float(total_value)


def _greedy_fill_value(weights, values, capacity, exclude_idx=None):
    """
    Compute total value from greedy 0/1 fill (take items by density if they fit).
    Optionally exclude one item index.
    """
    if capacity <= 0:
        return 0.0
    n = len(weights)
    if n == 0:
        return 0.0
    
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        w = weights[mask]
        v = values[mask]
    else:
        w = weights
        v = values
    
    if len(w) == 0:
        return 0.0
    
    valid = w > 0
    if not np.any(valid):
        return 0.0
    w = w[valid]
    v = v[valid]
    
    if len(w) == 0:
        return 0.0
    
    density = v / w
    order = np.argsort(density)[::-1]
    w_sorted = w[order]
    v_sorted = v[order]
    
    total_value = 0.0
    remaining_cap = capacity
    
    for i in range(len(w_sorted)):
        if w_sorted[i] <= remaining_cap:
            total_value += v_sorted[i]
            remaining_cap -= w_sorted[i]
    
    return float(total_value)


def _local_improve(weights, values, capacity, initial_sel, time_limit):
    """
    Given an initial selection (list of indices), try to improve via swaps.
    Returns the best first item to pick from the improved solution.
    """
    if not initial_sel:
        return -1
    
    n = len(weights)
    sel_set = set(initial_sel)
    cur_weight = sum(weights[i] for i in initial_sel)
    cur_value = sum(values[i] for i in initial_sel)
    
    # Precompute unselected feasible items
    unselected = [i for i in range(n) if i not in sel_set and 0 < weights[i] <= capacity]
    
    if not unselected:
        return initial_sel[0] if initial_sel else -1
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for out_idx in range(len(initial_sel)):
            if time.time() >= time_limit:
                break
            out_item = initial_sel[out_idx]
            out_w = weights[out_item]
            out_v = values[out_item]
            
            best_in = -1
            best_gain = 0
            for in_item in unselected:
                in_w = weights[in_item]
                in_v = values[in_item]
                if cur_weight - out_w + in_w <= capacity:
                    gain = in_v - out_v
                    if gain > best_gain:
                        best_gain = gain
                        best_in = in_item
            
            if best_in >= 0:
                initial_sel[out_idx] = best_in
                sel_set.remove(out_item)
                sel_set.add(best_in)
                cur_weight += weights[best_in] - out_w
                cur_value += values[best_in] - out_v
                unselected.remove(best_in)
                improved = True
    
    if not initial_sel:
        return -1
    return initial_sel[0]


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item for 0/1 knapsack using fractional surrogate lookahead.
    """
    global _START
    if _START is None:
        _START = time.time()
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        # Fallback: pick highest density feasible item
        feasible = np.where(remaining_weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        densities = remaining_values[feasible] / remaining_weights[feasible]
        return int(feasible[np.argmax(densities)])

    n = len(remaining_weights)
    if n == 0:
        return -1
    if remaining_capacity <= 0:
        return -1

    # Quick check: if only one item, take it if it fits
    if n == 1:
        if remaining_weights[0] <= remaining_capacity:
            return 0
        return -1

    # Compute densities
    densities = remaining_values / remaining_weights

    # Top-k candidates by density
    k = min(n, 40)
    top_k_idx = np.argsort(densities)[::-1][:k]

    # For each candidate, compute lookahead using fractional knapsack upper bound
    # This gives a tighter estimate of the potential value if we pick this item first
    best_score = -1.0
    best_idx = -1

    for i in top_k_idx:
        w_i = remaining_weights[i]
        v_i = remaining_values[i]
        if w_i > remaining_capacity:
            continue
        rem_cap = remaining_capacity - w_i
        # Fractional bound on remaining capacity after taking item i
        frac_val = _fractional_knapsack_value(remaining_weights, remaining_values, rem_cap, exclude_idx=int(i))
        total_val = v_i + frac_val
        if total_val > best_score:
            best_score = total_val
            best_idx = int(i)

    if best_idx < 0:
        # No feasible item found in top-k; try all feasible
        feasible = np.where(remaining_weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        return int(feasible[0])

    # Check deadline again
    if time.time() - _START > _DEADLINE:
        return best_idx

    # Local improvement: build a greedy solution starting with best_idx, then try swaps
    per_call_budget = 0.12
    local_deadline = time.time() + per_call_budget
    
    # Build initial greedy solution
    initial_sel = []
    used = set()
    cur_cap = remaining_capacity
    
    best_i = int(best_idx)
    if remaining_weights[best_i] <= cur_cap:
        initial_sel.append(best_i)
        used.add(best_i)
        cur_cap -= remaining_weights[best_i]
    
    # Fill rest greedily by density
    remaining_items = [i for i in range(n) if i not in used and remaining_weights[i] > 0]
    remaining_items.sort(key=lambda x: remaining_values[x] / remaining_weights[x], reverse=True)
    
    for i in remaining_items:
        if remaining_weights[i] <= cur_cap:
            initial_sel.append(i)
            used.add(i)
            cur_cap -= remaining_weights[i]
    
    if not initial_sel:
        return -1
    
    # Try local improvement with swaps
    improved_first = _local_improve(remaining_weights, remaining_values, remaining_capacity, initial_sel, local_deadline)
    
    if improved_first >= 0:
        return improved_first
    
    return best_idx
# EVOLVE-BLOCK-END

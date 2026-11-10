"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _fractional_upper_bound(capacity, weights, values):
    """Compute fractional knapsack upper bound for remaining items."""
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    feasible = weights <= capacity
    if not np.any(feasible):
        return 0.0
    
    w = weights[feasible]
    v = values[feasible]
    if len(w) == 0:
        return 0.0
    
    densities = v / w
    order = np.argsort(-densities)
    
    total_value = 0.0
    remaining_cap = capacity
    
    for idx in order:
        if remaining_cap <= 0:
            break
        if w[idx] <= remaining_cap:
            total_value += v[idx]
            remaining_cap -= w[idx]
        else:
            total_value += v[idx] * (remaining_cap / w[idx])
            break
    return total_value


def _greedy_complete(capacity, weights, values):
    """Complete greedily by value density; return total value."""
    if len(values) == 0 or capacity <= 0:
        return 0.0
    total = 0.0
    cap = capacity
    w = weights.copy()
    v = values.copy()
    n = len(w)
    for _ in range(n):
        if cap <= 0:
            break
        feasible = w <= cap
        if not np.any(feasible):
            break
        idxs = np.where(feasible)[0]
        if len(idxs) == 0:
            break
        densities = v[idxs] / w[idxs]
        best = idxs[np.argmax(densities)]
        total += v[best]
        cap -= w[best]
        w[best] = np.inf
        v[best] = 0.0
    return total


def _local_search_swap(capacity, weights, values, selected_mask, max_iters=2000, time_budget=None):
    """Try to improve solution by swapping selected items with unselected ones.
    Also tries adding unselected items if there's remaining capacity.
    Returns improved total value and updated selected mask."""
    total = float(np.sum(values[selected_mask]))
    cap_used = float(np.sum(weights[selected_mask]))
    remaining_cap = capacity - cap_used
    
    selected_idx = np.where(selected_mask)[0]
    unselected_idx = np.where(~selected_mask)[0]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        if time_budget is not None and time.time() - _START_TIME > time_budget:
            break
        improved = False
        iters += 1
        
        # First try adding items if there's remaining capacity
        if remaining_cap > 0 and len(unselected_idx) > 0:
            candidates = unselected_idx[weights[unselected_idx] <= remaining_cap]
            if len(candidates) > 0:
                # Pick the one with highest value density that fits
                dens = values[candidates] / weights[candidates]
                best_candidate = candidates[np.argmax(dens)]
                selected_mask[best_candidate] = True
                total += values[best_candidate]
                remaining_cap -= weights[best_candidate]
                selected_idx = np.where(selected_mask)[0]
                unselected_idx = np.where(~selected_mask)[0]
                improved = True
                continue
        
        # Try swapping each selected item with each unselected item
        for si in selected_idx:
            if improved:
                break
            w_out = weights[si]
            v_out = values[si]
            # New capacity after removing si: remaining_cap + w_out
            new_cap = remaining_cap + w_out
            # Find unselected items that fit in new_cap
            candidates = unselected_idx[weights[unselected_idx] <= new_cap]
            if len(candidates) == 0:
                continue
            # Choose the one that gives maximum value increase
            gains = values[candidates] - v_out
            max_gain = np.max(gains)
            if max_gain <= 0:
                continue
            best_candidate = candidates[np.argmax(gains)]
            # Perform swap
            selected_mask[si] = False
            selected_mask[best_candidate] = True
            total += max_gain
            remaining_cap = new_cap - weights[best_candidate]
            # Update index sets
            selected_idx = np.where(selected_mask)[0]
            unselected_idx = np.where(~selected_mask)[0]
            improved = True
    
    return total, selected_mask


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Pick the next item to add, or -1 to stop."""
    if remaining_capacity <= 0 or len(remaining_weights) == 0:
        return -1
    
    n = len(remaining_weights)
    
    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT - 5:
        return -1
    
    # Feasible items only
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_idx = np.where(feasible_mask)[0]
    f_w = remaining_weights[feasible_mask]
    f_v = remaining_values[feasible_mask]
    
    # Compute value density for feasible items
    densities = f_v / f_w
    
    # Base greedy: pick highest density
    best_idx_local = np.argmax(densities)
    best_idx_global = feasible_idx[best_idx_local]
    
    # Build initial solution by greedy
    selected = np.zeros(n, dtype=bool)
    cap = remaining_capacity
    w = remaining_weights.copy()
    v = remaining_values.copy()
    
    # Greedy fill - complete the knapsack
    for _ in range(n):
        if cap <= 0:
            break
        feas = w <= cap
        if not np.any(feas):
            break
        idxs = np.where(feas)[0]
        dens = v[idxs] / w[idxs]
        bi = idxs[np.argmax(dens)]
        selected[bi] = True
        cap -= w[bi]
        w[bi] = np.inf
        v[bi] = 0.0
    
    # Local search to improve - use more iterations with time budget
    time_budget = _TIME_LIMIT - 5
    if time.time() - _START_TIME < time_budget:
        total, selected = _local_search_swap(
            remaining_capacity, remaining_weights, remaining_values, 
            selected, max_iters=5000, time_budget=time_budget
        )
    
    # The first item in the improved solution is our pick
    # But we need to return an index into the remaining arrays
    # Find the first selected item in order of density preference
    if np.any(selected):
        # Prefer the highest density item among selected
        sel_idx = np.where(selected)[0]
        if len(sel_idx) > 0:
            # Pick the one with highest density among selected
            sel_densities = remaining_values[sel_idx] / remaining_weights[sel_idx]
            best_sel = sel_idx[np.argmax(sel_densities)]
            return int(best_sel)
    
    # Fallback to base greedy
    return int(best_idx_global)
# EVOLVE-BLOCK-END

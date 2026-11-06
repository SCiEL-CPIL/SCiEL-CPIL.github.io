"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None


def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _lp_relaxation_upper_bound(capacity, weights, values):
    """
    Compute the LP relaxation upper bound for 0/1 knapsack.
    Sort by value/weight density, take items fully until capacity is
    nearly exhausted, then take a fractional last item.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    feasible = weights <= capacity + 1e-12
    if not np.any(feasible):
        return 0.0
    
    w = weights[feasible]
    v = values[feasible]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    
    order = np.argsort(-densities)
    
    total_value = 0.0
    total_weight = 0.0
    
    for idx in order:
        if total_weight + w[idx] <= capacity + 1e-12:
            total_value += v[idx]
            total_weight += w[idx]
        else:
            remaining = capacity - total_weight
            if remaining > 1e-12:
                frac = remaining / w[idx]
                total_value += frac * v[idx]
            break
    
    return total_value


def _greedy_fill(capacity, weights, values, density_power=1.0):
    """Fill remaining capacity using greedy by value/weight^power."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    feasible = weights <= capacity + 1e-12
    if not np.any(feasible):
        return 0.0, []
    
    w = weights[feasible]
    v = values[feasible]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / np.power(w, density_power)
    
    order = np.argsort(-densities)
    
    total_value = 0.0
    total_weight = 0.0
    picked = []
    
    for idx in order:
        if w[idx] <= capacity - total_weight + 1e-12:
            total_value += v[idx]
            total_weight += w[idx]
            picked.append(idx)
        if total_weight >= capacity - 1e-12:
            break
    
    return total_value, picked


def _solve_lp_with_lambda(capacity, weights, values, max_iter=50, tol=1e-8):
    """
    Solve the LP relaxation using a dual/Lagrangian approach.
    We binary search on lambda (shadow price) such that the sum of
    weights of items with v_i - lambda * w_i > 0 approximately equals capacity.
    
    Returns: (lambda, total_value_of_full_items, fractional_value)
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, 0.0, 0.0
    
    feasible = weights <= capacity + 1e-12
    if not np.any(feasible):
        return 0.0, 0.0, 0.0
    
    w = weights[feasible]
    v = values[feasible]
    
    # Find range for lambda
    # lambda > 0: items with v/w > lambda are taken fully
    # We want sum(w_i) for v_i/w_i > lambda to be close to capacity
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    
    # Sort by density descending
    order = np.argsort(-densities)
    w_sorted = w[order]
    v_sorted = v[order]
    d_sorted = densities[order]
    
    # Binary search on lambda
    lo = 0.0
    hi = d_sorted[0] + 1.0  # max density + epsilon
    
    # Find the lambda where cumulative weight crosses capacity
    # For a given lambda, take all items with density > lambda
    # The lambda that makes sum(w) = capacity is what we want
    
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        
        # Items with density > mid
        mask = d_sorted > mid
        cum_weight = np.sum(w_sorted[mask])
        
        if cum_weight < capacity:
            # Need lower lambda to include more items
            lo = mid
        else:
            # Need higher lambda to exclude items
            hi = mid
        
        if hi - lo < tol:
            break
    
    lambda_val = (lo + hi) / 2.0
    
    # Now compute the LP solution with this lambda
    # Take all items with v_i - lambda*w_i > 0
    reduced_costs = v - lambda_val * w
    
    # Items with positive reduced cost are taken fully
    full_mask = reduced_costs > 1e-10
    full_weight = np.sum(w[full_mask])
    full_value = np.sum(v[full_mask])
    
    # If there's remaining capacity, take a fractional item
    remaining_cap = capacity - full_weight
    if remaining_cap > 1e-10:
        # Find the item with the best density among those not fully taken
        # that fits fractionally
        # Actually, in LP relaxation, we take the item at the "break" point
        # Let's find items with density close to lambda
        # The fractional item is the one where density is just below lambda
        # or we can just find the best remaining item
        
        # Find items not in full_mask
        not_full_mask = ~full_mask
        if np.any(not_full_mask):
            w_rem = w[not_full_mask]
            v_rem = v[not_full_mask]
            if len(w_rem) > 0:
                with np.errstate(divide='ignore', invalid='ignore'):
                    d_rem = v_rem / w_rem
                best_idx = np.argmax(d_rem)
                frac = min(1.0, remaining_cap / w_rem[best_idx])
                fractional_value = frac * v_rem[best_idx]
            else:
                fractional_value = 0.0
        else:
            fractional_value = 0.0
    else:
        fractional_value = 0.0
    
    total_value = full_value + fractional_value
    return lambda_val, full_value, fractional_value


def _estimate_total_value(remaining_capacity, remaining_weights, remaining_values):
    """
    Estimate the total value achievable from the current state by:
    1. Computing the LP relaxation upper bound
    2. Trying each feasible item as the first pick
    3. For each first pick, using LP relaxation on the remainder
    4. Combining with greedy strategies
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return 0.0, -1
    
    # Feasible items
    feasible_mask = remaining_weights <= remaining_capacity + 1e-12
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return 0.0, -1
    
    best_estimate = 0.0
    best_first_pick = -1
    
    # Compute LP relaxation upper bound for the full set
    full_lp_bound = _lp_relaxation_upper_bound(remaining_capacity, remaining_weights, remaining_values)
    
    # Get densities for sorting
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = remaining_values[feasible_indices] / remaining_weights[feasible_indices]
    
    # Sort feasible items by density
    sorted_feasible = feasible_indices[np.argsort(-densities)]
    
    # Evaluate top K candidates as first pick
    K = min(len(sorted_feasible), max(10, n // 5))
    top_candidates = sorted_feasible[:K]
    
    # Also include some low-weight items
    if n > 20:
        light_items = feasible_indices[np.argsort(remaining_weights[feasible_indices])[:max(3, n // 20)]]
        top_candidates = np.unique(np.concatenate([top_candidates, light_items]))
    
    for first_idx in top_candidates:
        wi = remaining_weights[first_idx]
        vi = remaining_values[first_idx]
        cap_rem = remaining_capacity - wi
        
        if cap_rem < -1e-12:
            continue
        
        # Create remaining arrays without first_idx
        mask = np.ones(n, dtype=bool)
        mask[first_idx] = False
        rem_w = remaining_weights[mask]
        rem_v = remaining_values[mask]
        
        # LP relaxation upper bound for the remainder
        lp_bound_rem = _lp_relaxation_upper_bound(cap_rem, rem_w, rem_v)
        
        # Greedy fill for a lower-bound estimate
        greedy_fill_value, _ = _greedy_fill(cap_rem, rem_w, rem_v, density_power=1.0)
        
        # Combine: weighted blend
        combined_rem = 0.7 * lp_bound_rem + 0.3 * greedy_fill_value
        
        total = vi + combined_rem
        if total > best_estimate:
            best_estimate = total
            best_first_pick = int(first_idx)
    
    # Also try pure greedy strategies
    pure_greedy_value, _ = _greedy_fill(remaining_capacity, remaining_weights, remaining_values, density_power=1.0)
    sqrt_greedy_value, _ = _greedy_fill(remaining_capacity, remaining_weights, remaining_values, density_power=0.5)
    
    if best_first_pick == -1:
        best_estimate = max(pure_greedy_value, sqrt_greedy_value)
        with np.errstate(divide='ignore', invalid='ignore'):
            densities_all = remaining_values / remaining_weights
        densities_all[~feasible_mask] = -np.inf
        best_first_pick = int(np.argmax(densities_all))
        if not feasible_mask[best_first_pick]:
            best_first_pick = -1
    else:
        if pure_greedy_value > best_estimate:
            with np.errstate(divide='ignore', invalid='ignore'):
                densities_all = remaining_values / remaining_weights
            densities_all[~feasible_mask] = -np.inf
            first_in_greedy = int(np.argmax(densities_all))
            if feasible_mask[first_in_greedy]:
                best_first_pick = first_in_greedy
                best_estimate = pure_greedy_value
        if sqrt_greedy_value > best_estimate:
            with np.errstate(divide='ignore', invalid='ignore'):
                densities_sqrt = remaining_values / np.sqrt(remaining_weights)
            densities_sqrt[~feasible_mask] = -np.inf
            first_in_sqrt = int(np.argmax(densities_sqrt))
            if feasible_mask[first_in_sqrt]:
                best_first_pick = first_in_sqrt
                best_estimate = sqrt_greedy_value
    
    return best_estimate, best_first_pick


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check for feasible items
    feasible_mask = remaining_weights <= remaining_capacity + 1e-12
    if not np.any(feasible_mask):
        return -1
    
    # Deadline guard
    elapsed = _get_elapsed()
    if elapsed > 135.0:
        # Fallback to simple greedy
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = remaining_values / remaining_weights
        densities[~feasible_mask] = -np.inf
        return int(np.argmax(densities))
    
    # Estimate total value and pick the best first item
    estimate, best_pick = _estimate_total_value(remaining_capacity, remaining_weights, remaining_values)
    
    if best_pick == -1:
        # Fallback
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = remaining_values / remaining_weights
        densities[~feasible_mask] = -np.inf
        best_pick = int(np.argmax(densities))
        if not feasible_mask[best_pick]:
            return -1
    
    return best_pick
# EVOLVE-BLOCK-END

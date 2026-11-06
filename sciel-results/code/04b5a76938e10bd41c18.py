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
    This is a tight upper bound and a smooth surrogate for the true optimum.
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


def _estimate_total_value(remaining_capacity, remaining_weights, remaining_values):
    """
    Estimate the total value achievable from the current state by:
    1. Trying each feasible item as the first pick
    2. For each first pick, using LP relaxation upper bound + greedy fill
       to estimate the remaining value (LP gives a tighter, smoother bound)
    3. Combining with alternative greedy strategies
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
    
    # Score candidates using a combination of density and LP-based lookahead
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = remaining_values[feasible_indices] / remaining_weights[feasible_indices]
    
    # Sort feasible items by density
    sorted_feasible = feasible_indices[np.argsort(-densities)]
    
    # Evaluate top K candidates as first pick
    K = min(len(sorted_feasible), max(15, n // 4))
    top_candidates = sorted_feasible[:K]
    
    # Also include some low-weight items (they might be good first picks)
    if n > 20:
        light_items = feasible_indices[np.argsort(remaining_weights[feasible_indices])[:max(5, n // 15)]]
        top_candidates = np.unique(np.concatenate([top_candidates, light_items]))
    
    # Pre-compute LP bound for the full set (used as baseline)
    full_lp_bound = _lp_relaxation_upper_bound(remaining_capacity, remaining_weights, remaining_values)
    
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
        
        # LP relaxation upper bound for the remainder (tighter than greedy)
        lp_bound_rem = _lp_relaxation_upper_bound(cap_rem, rem_w, rem_v)
        
        # Greedy fill for a lower-bound estimate
        greedy_fill_value, _ = _greedy_fill(cap_rem, rem_w, rem_v, density_power=1.0)
        
        # Combine: use a weighted blend of LP bound (upper) and greedy (lower)
        # LP bound is an upper bound, greedy is a lower bound.
        # Use LP bound as primary estimate since it's tighter and smoother.
        # Blend: 0.7 * LP + 0.3 * greedy to avoid over-optimism
        combined_rem = 0.7 * lp_bound_rem + 0.3 * greedy_fill_value
        
        total = vi + combined_rem
        if total > best_estimate:
            best_estimate = total
            best_first_pick = int(first_idx)
    
    # Also try pure greedy without committing to a specific first item
    pure_greedy_value, _ = _greedy_fill(remaining_capacity, remaining_weights, remaining_values, density_power=1.0)
    
    # Try sqrt-density greedy
    sqrt_greedy_value, _ = _greedy_fill(remaining_capacity, remaining_weights, remaining_values, density_power=0.5)
    
    # LP bound for the full set as a reference
    # If the best first-pick estimate is close to the LP bound, it's likely good
    
    # Combine estimates
    if best_first_pick == -1:
        best_estimate = max(pure_greedy_value, sqrt_greedy_value)
        # Pick the item that greedy would pick first
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = remaining_values / remaining_weights
        densities[~feasible_mask] = -np.inf
        best_first_pick = int(np.argmax(densities))
        if not feasible_mask[best_first_pick]:
            best_first_pick = -1
    else:
        # Use the max of the lookahead estimate and pure greedy
        # But keep the first pick from lookahead if it's better
        if pure_greedy_value > best_estimate:
            # Pure greedy is better; use its first pick
            with np.errstate(divide='ignore', invalid='ignore'):
                densities = remaining_values / remaining_weights
            densities[~feasible_mask] = -np.inf
            first_in_greedy = int(np.argmax(densities))
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
    if elapsed > 130.0:
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

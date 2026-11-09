"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # 5s safety margin before 120s limit


def _elapsed_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _lp_bound(capacity, weights, values, order):
    """Compute fractional knapsack upper bound using density order."""
    cap = capacity
    val = 0.0
    for i in order:
        w = weights[i]
        if w <= cap:
            cap -= w
            val += values[i]
        else:
            val += values[i] * (cap / w)
            break
    return val


def _simulate_total(capacity, weights, values, order, skip_indices):
    """Simulate greedy fill following `order`, skipping items in `skip_indices`. Return total value."""
    if isinstance(skip_indices, (int, np.integer)):
        skip_set = {skip_indices}
    else:
        skip_set = set(skip_indices)
    cap = capacity
    val = 0.0
    for i in order:
        if i in skip_set:
            continue
        w = weights[i]
        if w <= cap:
            cap -= w
            val += values[i]
        if cap <= 0:
            break
    return val


def _discrete_completion(capacity, weights, values, order, skip_indices):
    """
    Estimate best discrete completion value after skipping given indices.
    Uses greedy density fill but also considers a few alternative orderings
    to get a tighter lower bound on achievable value.
    """
    if isinstance(skip_indices, (int, np.integer)):
        skip_set = {skip_indices}
    elif skip_indices is None:
        skip_set = set()
    else:
        skip_set = set(skip_indices)
    
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Primary: density greedy
    best_val = _simulate_total(capacity, weights, values, order, skip_set)
    
    # Secondary: value-first greedy for items that fit
    if not _elapsed_ok():
        return best_val
    
    # Try a value/weight^0.5 ordering (modified density)
    try:
        mod_densities = values / np.sqrt(weights)
        mod_order = np.argsort(-mod_densities)
        alt_val = _simulate_total(capacity, weights, values, mod_order, skip_set)
        if alt_val > best_val:
            best_val = alt_val
    except:
        pass
    
    if not _elapsed_ok():
        return best_val
    
    # Try pure value ordering
    try:
        val_order = np.argsort(-values)
        alt_val2 = _simulate_total(capacity, weights, values, val_order, skip_set)
        if alt_val2 > best_val:
            best_val = alt_val2
    except:
        pass
    
    return best_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # If time is running out, fall back to simple density greedy
    if not _elapsed_ok():
        return int(np.argmax(remaining_values / remaining_weights))

    # Convert to numpy for speed
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    capacity = float(remaining_capacity)

    # Compute density-based order
    densities = values / weights
    order = np.argsort(-densities)  # descending density

    # Compute the LP (fractional knapsack) upper bound
    lp_ub = _lp_bound(capacity, weights, values, order)

    # For each feasible item, compute:
    # 1. The value if we pick it: values[i] + discrete_completion(cap - w_i, ..., skip={i})
    # 2. The LP bound after picking it: values[i] + lp_bound(cap - w_i, ..., skip={i})
    # 3. The gap: lp_bound_after - discrete_completion_after
    # We want to minimize the gap (maximize the discrete completion relative to LP bound)
    # i.e., pick the item that makes the remaining problem closest to its LP bound

    candidates = []  # list of (gap, total_estimate, idx)
    feasible_indices = []
    
    for i in order:
        w_i = weights[i]
        if w_i > capacity:
            continue
        feasible_indices.append(i)
    
    if not feasible_indices:
        return -1

    # Evaluate each feasible candidate
    for i in feasible_indices:
        if not _elapsed_ok():
            break
        
        w_i = weights[i]
        v_i = values[i]
        cap_after = capacity - w_i
        
        if cap_after <= 0:
            # Picking this item fills the knapsack completely
            total_est = v_i
            lp_after = 0.0
            gap = 0.0
        else:
            # LP bound after picking i
            lp_after = _lp_bound(cap_after, weights, values, order)
            # Discrete completion estimate after picking i
            disc_after = _discrete_completion(cap_after, weights, values, order, i)
            total_est = v_i + disc_after
            gap = max(0.0, (v_i + lp_after) - total_est)
        
        candidates.append((gap, total_est, i))

    if not candidates:
        return -1

    # Sort by gap ascending (prefer items that minimize the LP gap)
    # Break ties by total_est descending
    candidates.sort(key=lambda x: (x[0], -x[1]))

    # If time permits, do a 2-step lookahead refinement on the top few candidates
    elapsed = time.time() - _START_TIME
    remaining_time = _DEADLINE - elapsed
    
    if remaining_time > 5.0:
        K = min(3, len(candidates))
        best_final_score = -1.0
        best_final_idx = candidates[0][2]
        
        for rank in range(K):
            if not _elapsed_ok():
                break
            _, total_est, idx = candidates[rank]
            
            # 2-step lookahead: pick idx, then pick best j, then greedy
            w1 = weights[idx]
            cap1 = capacity - w1
            v1 = values[idx]
            
            if cap1 <= 0:
                sim_val = v1
            else:
                best_total = v1
                for j in order:
                    if j == idx:
                        continue
                    w2 = weights[j]
                    if w2 > cap1:
                        continue
                    # Simulate greedy after picking idx and j
                    sim_val = v1 + values[j] + _simulate_total(
                        cap1 - w2,
                        weights,
                        values,
                        order,
                        (idx, j)
                    )
                    if sim_val > best_total:
                        best_total = sim_val
                sim_val = best_total
            
            if sim_val > best_final_score:
                best_final_score = sim_val
                best_final_idx = idx
        
        if best_final_score >= 0:
            return int(best_final_idx)

    # Return the best candidate by LP gap criterion
    return int(candidates[0][2])
# EVOLVE-BLOCK-END

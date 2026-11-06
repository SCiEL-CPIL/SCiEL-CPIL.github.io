"""0/1 Knapsack heuristic: lookahead with Lagrangian-guided completion and top-K candidate pruning."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 112.0  # safety margin before 120s limit


def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))


def _find_optimal_lambda(weights, values, capacity, max_iter=50, tol=1e-6):
    """
    Find lambda (shadow price) such that the Lagrangian relaxation
    sum(v_i - lambda * w_i for items where v_i - lambda * w_i > 0)
    approximately exhausts the capacity.
    
    Uses bisection on lambda to find the dual variable that makes
    the fractional knapsack optimal.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    
    # Find feasible range for lambda
    # lambda_min: smallest lambda where all items have positive reduced cost
    # lambda_max: largest lambda where no items have positive reduced cost
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(w > 0, v / w, np.inf)
    
    valid = w > 0
    if not np.any(valid):
        return 0.0
    
    # lambda where item i has zero reduced cost: lambda = v_i / w_i
    # We want to find lambda such that sum(w_i for v_i - lambda*w_i > 0) ~ capacity
    
    lo = 0.0
    hi = np.max(densities[valid]) + 1.0
    
    # If even at lambda=0 we can't fill capacity, return 0
    total_w_at_zero = np.sum(w[valid])
    if total_w_at_zero <= capacity:
        return 0.0
    
    # Bisection to find lambda
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        # Items with positive reduced cost at this lambda
        reduced_costs = v - mid * w
        positive_mask = (reduced_costs > 0) & (w > 0)
        total_w = np.sum(w[positive_mask])
        
        if total_w > capacity:
            # Need higher lambda to exclude more items
            lo = mid
        else:
            # Need lower lambda to include more items
            hi = mid
        
        if hi - lo < tol:
            break
    
    return (lo + hi) / 2.0


def _completion_value(remaining_capacity, weights, values, exclude_idx=None):
    """
    Estimate achievable 0/1 value for the remaining items:
    1) Lagrangian relaxation to find optimal lambda (shadow price)
    2) Greedy fill using items with positive reduced cost (v_i - lambda * w_i > 0)
    3) One pass of 1-swap local refinement
    Optionally exclude one item (the candidate being evaluated).
    Returns the estimated total value of the completion (excluding the candidate).
    """
    if remaining_capacity <= 0:
        return 0.0

    n = len(weights)
    if n == 0:
        return 0.0

    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
    else:
        mask = np.ones(n, dtype=bool)

    # Only items that individually fit are candidates for the greedy fill
    mask &= (weights <= remaining_capacity)
    if not np.any(mask):
        return 0.0

    idx = np.where(mask)[0]
    w = weights[idx]
    v = values[idx]
    m = len(idx)

    # Find optimal lambda via Lagrangian relaxation
    lambda_star = _find_optimal_lambda(w, v, remaining_capacity)
    
    # Items with positive reduced cost are preferred
    reduced_costs = v - lambda_star * w
    positive_mask = reduced_costs > 0
    
    # Sort by reduced cost descending (equivalent to density when lambda is optimal)
    order = np.argsort(-reduced_costs)
    w_sorted = w[order]
    v_sorted = v[order]
    rc_sorted = reduced_costs[order]
    
    # Greedy 0/1 fill using Lagrangian-guided order
    chosen = np.zeros(m, dtype=bool)
    cap = remaining_capacity
    total_val = 0.0
    for j in range(m):
        if w_sorted[j] <= cap:
            chosen[order[j]] = True
            total_val += v_sorted[j]
            cap -= w_sorted[j]
            if cap <= 0:
                break

    # 1-swap local refinement: try replacing a chosen item with an unchosen feasible item
    chosen_pos = np.where(chosen)[0]
    unchosen_mask = (~chosen) & (w <= remaining_capacity)
    unchosen_pos = np.where(unchosen_mask)[0]

    if len(chosen_pos) > 0 and len(unchosen_pos) > 0:
        # For efficiency, only consider top unchosen by reduced cost (they are the best swaps)
        uc_w = w[unchosen_pos]
        uc_v = v[unchosen_pos]
        uc_rc = reduced_costs[unchosen_pos]
        uc_order = np.argsort(-uc_rc)
        # Limit swaps to top ~30 unchosen for speed
        max_uc = min(len(uc_order), 30)
        uc_order = uc_order[:max_uc]

        for c in chosen_pos:
            wc = w[c]
            vc = v[c]
            for uo in uc_order:
                u = unchosen_pos[uo]
                wu = w[u]
                vu = v[u]
                # Swapping: remove c (free wc, lose vc), add u (use wu, gain vu)
                # Feasible if wu <= wc (since we free wc capacity)
                if wu <= wc and (vu - vc) > 0:
                    # Apply swap
                    total_val += (vu - vc)
                    chosen[c] = False
                    chosen[u] = True
                    break  # one swap per chosen item per pass

    return float(total_val)


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item using lookahead: for each top-K candidate by density,
    estimate total value = candidate value + greedy+swap completion of remaining.
    """
    # Deadline guard
    if _remaining_budget() < 0.5:
        # Fallback: simple density greedy
        w = np.asarray(remaining_weights, dtype=np.float64)
        v = np.asarray(remaining_values, dtype=np.float64)
        cap = float(remaining_capacity)
        if len(w) == 0 or cap <= 0:
            return -1
        feasible = w <= cap
        if not np.any(feasible):
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(w > 0, v / w, -np.inf)
        density[~feasible] = -np.inf
        best = int(np.argmax(density))
        return best if feasible[best] else -1

    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)
    n = len(w)

    if n == 0 or cap <= 0:
        return -1

    # Find feasible items
    feasible_mask = w <= cap
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]

    # Compute Lagrangian-guided scores: find lambda for current state
    lambda_star = _find_optimal_lambda(w[feasible_mask], v[feasible_mask], cap, max_iter=30, tol=1e-5)
    
    # Reduced cost as selection score
    reduced_costs = v - lambda_star * w
    reduced_costs[~feasible_mask] = -np.inf

    # Sort feasible indices by reduced cost descending
    feasible_rc = reduced_costs[feasible_indices]
    sorted_order = np.argsort(-feasible_rc)
    sorted_feasible = feasible_indices[sorted_order]

    # Adaptive K: more candidates when more time available
    budget = _remaining_budget()
    if budget > 30.0:
        K = min(len(sorted_feasible), 18)
    elif budget > 15.0:
        K = min(len(sorted_feasible), 12)
    elif budget > 8.0:
        K = min(len(sorted_feasible), 8)
    else:
        K = min(len(sorted_feasible), 5)

    best_idx = -1
    best_estimated_value = -1.0

    for i in range(K):
        idx = sorted_feasible[i]
        wi = w[idx]
        vi = v[idx]

        rem_cap = cap - wi
        if rem_cap < 0:
            continue

        # Completion estimate (Lagrangian-guided greedy + swap), excluding this candidate
        completion_val = _completion_value(rem_cap, w, v, exclude_idx=idx)
        total_est = vi + completion_val

        if total_est > best_estimated_value:
            best_estimated_value = total_est
            best_idx = idx

        # Time check inside loop
        if _remaining_budget() < 1.0:
            break

    return int(best_idx) if best_idx >= 0 else -1
# EVOLVE-BLOCK-END

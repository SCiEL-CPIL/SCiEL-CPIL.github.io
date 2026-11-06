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
    
    valid = w > 0
    if not np.any(valid):
        return 0.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(valid, v / w, np.inf)
    
    lo = 0.0
    hi = np.max(densities[valid]) + 1.0
    
    total_w_at_zero = np.sum(w[valid])
    if total_w_at_zero <= capacity:
        return 0.0
    
    # Bisection to find lambda
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        reduced_costs = v - mid * w
        positive_mask = (reduced_costs > 0) & valid
        total_w = np.sum(w[positive_mask])
        
        if total_w > capacity:
            lo = mid
        else:
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

    # Local refinement: 1-swap then 2-swap passes
    chosen_pos = np.where(chosen)[0]
    unchosen_mask = (~chosen) & (w <= remaining_capacity)
    unchosen_pos = np.where(unchosen_mask)[0]

    if len(chosen_pos) > 0 and len(unchosen_pos) > 0:
        uc_w = w[unchosen_pos]
        uc_v = v[unchosen_pos]
        uc_rc = reduced_costs[unchosen_pos]
        uc_order = np.argsort(-uc_rc)
        max_uc = min(len(uc_order), 40)
        uc_order = uc_order[:max_uc]

        # Pass 1: 1-swap (replace one chosen with one unchosen)
        for c in chosen_pos:
            if not chosen[c]:
                continue
            wc = w[c]
            vc = v[c]
            best_gain = 0.0
            best_u = -1
            for uo in uc_order:
                u = unchosen_pos[uo]
                if chosen[u]:
                    continue
                wu = w[u]
                vu = v[u]
                if wu <= wc:
                    gain = vu - vc
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
            if best_u >= 0:
                total_val += best_gain
                chosen[c] = False
                chosen[best_u] = True

        # Pass 2: 2-swap (replace two chosen with two unchosen)
        chosen_pos2 = np.where(chosen)[0]
        unchosen_mask2 = (~chosen) & (w <= remaining_capacity)
        unchosen_pos2 = np.where(unchosen_mask2)[0]
        if len(chosen_pos2) >= 2 and len(unchosen_pos2) >= 2:
            uc2_w = w[unchosen_pos2]
            uc2_v = v[unchosen_pos2]
            uc2_rc = reduced_costs[unchosen_pos2]
            uc2_order = np.argsort(-uc2_rc)
            max_uc2 = min(len(uc2_order), 25)
            uc2_order = uc2_order[:max_uc2]
            max_c2 = min(len(chosen_pos2), 25)

            best_2swap_gain = 0.0
            best_2swap = None
            for ci in range(max_c2):
                c1 = chosen_pos2[ci]
                wc1 = w[c1]
                vc1 = v[c1]
                for cj in range(ci + 1, max_c2):
                    c2 = chosen_pos2[cj]
                    wc2 = w[c2]
                    vc2 = v[c2]
                    freed = wc1 + wc2
                    lost = vc1 + vc2
                    for ui in uc2_order:
                        u1 = unchosen_pos2[ui]
                        wu1 = w[u1]
                        vu1 = v[u1]
                        if wu1 > freed:
                            continue
                        rem = freed - wu1
                        for uj in uc2_order:
                            if uj == ui:
                                continue
                            u2 = unchosen_pos2[uj]
                            wu2 = w[u2]
                            vu2 = v[u2]
                            if wu2 > rem:
                                continue
                            gain = (vu1 + vu2) - lost
                            if gain > best_2swap_gain:
                                best_2swap_gain = gain
                                best_2swap = (c1, c2, u1, u2)
            if best_2swap is not None:
                c1, c2, u1, u2 = best_2swap
                total_val += best_2swap_gain
                chosen[c1] = False
                chosen[c2] = False
                chosen[u1] = True
                chosen[u2] = True

    # Final high-precision polish: re-derive lambda at tighter tolerance and re-greedy
    if remaining_capacity > 0 and len(chosen) > 0:
        lambda_refined = _find_optimal_lambda(w, v, remaining_capacity, max_iter=60, tol=1e-9)
        rc_refined = v - lambda_refined * w
        order_refined = np.argsort(-rc_refined)
        cap2 = remaining_capacity
        total_val2 = 0.0
        chosen2 = np.zeros(m, dtype=bool)
        for j in range(m):
            jj = order_refined[j]
            if w[jj] <= cap2:
                chosen2[jj] = True
                total_val2 += v[jj]
                cap2 -= w[jj]
                if cap2 <= 0:
                    break
        if total_val2 > total_val:
            total_val = total_val2
            chosen = chosen2

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
    lambda_star = _find_optimal_lambda(w[feasible_mask], v[feasible_mask], cap, max_iter=40, tol=1e-7)
    
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
        K = min(len(sorted_feasible), 20)
    elif budget > 15.0:
        K = min(len(sorted_feasible), 14)
    elif budget > 8.0:
        K = min(len(sorted_feasible), 10)
    else:
        K = min(len(sorted_feasible), 6)

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

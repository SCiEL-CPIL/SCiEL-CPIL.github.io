# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 3.0


def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)


def _lp_bound_gap(capacity, weights, values, order_indices, candidate_idx):
    """
    Compute the fractional knapsack bound when candidate_idx is taken first,
    then greedily fill the rest in density order. Returns (total_value, gap_estimate).
    The gap is estimated as the value of the fractional last item that would need
    to be rounded down, giving a proxy for how "close to LP optimal" this choice is.
    """
    sim_cap = capacity - weights[candidate_idx]
    sim_val = values[candidate_idx]
    if sim_cap < 0:
        return 0.0, 0.0

    # Fill remaining with density order, skipping candidate
    for idx in order_indices:
        if idx == candidate_idx:
            continue
        w = weights[idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += values[idx]
        else:
            # Fractional item - this is where the gap lives
            # The gap is roughly the fractional value we can't take
            frac_val = values[idx] * (sim_cap / w) if w > 0 else 0.0
            return sim_val + frac_val, frac_val
    return sim_val, 0.0


def _compute_lp_optimal_order(weights, values, capacity):
    """
    Compute the LP optimal fractional knapsack solution.
    Returns the order indices and the set of fully-taken items.
    """
    n = len(weights)
    if n == 0:
        return np.array([], dtype=int), set()
    
    densities = values / np.where(weights > 0, weights, 1.0)
    order = np.argsort(-densities)
    
    sim_cap = capacity
    fully_taken = set()
    for idx in order:
        if weights[idx] <= sim_cap:
            sim_cap -= weights[idx]
            fully_taken.add(idx)
        else:
            break
    return order, fully_taken


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    if not _elapsed_ok():
        # Fallback: highest density feasible item
        feasible = remaining_weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        idxs = np.where(feasible)[0]
        densities = remaining_values[idxs] / np.where(remaining_weights[idxs] > 0, remaining_weights[idxs], 1.0)
        return int(idxs[np.argmax(densities)])

    # Filter to feasible items only
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    feasible_idx = np.where(feasible_mask)[0]
    f_w = remaining_weights[feasible_idx]
    f_v = remaining_values[feasible_idx]
    m = len(f_w)

    # Density order for the feasible set (local indices)
    safe_w = np.where(f_w > 0, f_w, 1.0)
    densities = f_v / safe_w
    order = np.argsort(-densities)  # local indices, best density first

    # Compute LP optimal fractional solution
    lp_order, lp_fully_taken = _compute_lp_optimal_order(f_w, f_v, remaining_capacity)
    
    # Strategy: prioritize items that are in the LP fully-taken set
    # Among those, prefer higher density. For items not in LP set,
    # use a surrogate score based on how close to the LP bound they bring us.
    
    # First tier: items in LP fully-taken set
    lp_local_indices = [i for i in lp_fully_taken if i < m]
    
    if len(lp_local_indices) > 0:
        # Among LP-taken items, pick the one with highest density
        lp_arr = np.array(lp_local_indices)
        lp_densities = densities[lp_arr]
        best_lp_local = lp_arr[np.argmax(lp_densities)]
        
        # But also check: is there an item NOT in LP set that gives a better
        # 1-step lookahead value? Use LP gap as a tiebreaker/secondary score.
        # For a few top candidates, simulate and compare.
        
        # Candidates: best LP item + top few non-LP items by density
        non_lp_order = [i for i in order if i not in lp_fully_taken]
        K_non_lp = min(10, len(non_lp_order))
        non_lp_candidates = non_lp_order[:K_non_lp]
        
        all_candidates = [best_lp_local] + non_lp_candidates
        
        # Evaluate each candidate: simulate greedy fill after picking it
        best_score = -1.0
        best_local = best_lp_local
        full_order_list = order.tolist()
        
        for cand in all_candidates:
            # Simulate: pick cand first, then greedy fill
            sim_cap = remaining_capacity - f_w[cand]
            sim_val = f_v[cand]
            if sim_cap < 0:
                continue
            for idx in full_order_list:
                if idx == cand:
                    continue
                w = f_w[idx]
                if w <= sim_cap:
                    sim_cap -= w
                    sim_val += f_v[idx]
                if sim_cap <= 0:
                    break
            
            # Bonus for items in LP set (they're "closer to optimal")
            lp_bonus = 0.0
            if cand in lp_fully_taken:
                lp_bonus = 0.01 * sim_val  # small bonus to prefer LP items
            
            score = sim_val + lp_bonus
            if score > best_score:
                best_score = score
                best_local = cand
            if not _elapsed_ok():
                break
    else:
        # No LP fully-taken items (shouldn't happen if capacity > 0),
        # fall back to 1-step lookahead over top density items
        K = min(30, m)
        candidates = order[:K]
        full_order_list = order.tolist()
        
        best_val = -1.0
        best_local = int(candidates[0]) if len(candidates) > 0 else 0
        for cand in candidates:
            sim_cap = remaining_capacity - f_w[cand]
            sim_val = f_v[cand]
            if sim_cap < 0:
                continue
            for idx in full_order_list:
                if idx == cand:
                    continue
                w = f_w[idx]
                if w <= sim_cap:
                    sim_cap -= w
                    sim_val += f_v[idx]
                if sim_cap <= 0:
                    break
            if sim_val > best_val:
                best_val = sim_val
                best_local = int(cand)
            if not _elapsed_ok():
                break

    # Map local index back to index in remaining arrays
    return int(feasible_idx[best_local])
# EVOLVE-BLOCK-END

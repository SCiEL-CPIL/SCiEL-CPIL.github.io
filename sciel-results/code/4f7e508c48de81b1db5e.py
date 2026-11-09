"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 3.0


def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)


def _simulate_greedy(capacity, weights, values, order_indices, exclude_idx):
    """Simulate greedy fill over order_indices, skipping exclude_idx. Return total value."""
    sim_cap = capacity
    sim_val = 0.0
    for idx in order_indices:
        if idx == exclude_idx:
            continue
        w = weights[idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += values[idx]
        if sim_cap <= 0:
            break
    return sim_val


def _simulate_greedy_accurate(capacity, weights, values, order_indices, exclude_idx):
    """
    More accurate simulation: after picking exclude_idx first, greedily pick the
    best-density feasible item at each step (recomputing feasibility).
    Returns (total_value, list_of_picked_indices).
    """
    sim_cap = capacity
    sim_val = 0.0
    picked = []
    
    # First, pick the candidate (exclude_idx)
    w = weights[exclude_idx]
    if w <= sim_cap:
        sim_cap -= w
        sim_val += values[exclude_idx]
        picked.append(exclude_idx)
    
    # Now greedily fill remaining capacity using density order, but only feasible items
    # We need to track which items are still available
    available = set(order_indices)
    available.discard(exclude_idx)
    
    while sim_cap > 0 and available:
        # Find the best-density feasible item among available
        best_idx = -1
        best_density = -1.0
        for idx in available:
            w = weights[idx]
            if w <= sim_cap:
                d = values[idx] / w
                if d > best_density:
                    best_density = d
                    best_idx = idx
        if best_idx < 0:
            break
        sim_cap -= weights[best_idx]
        sim_val += values[best_idx]
        picked.append(best_idx)
        available.discard(best_idx)
    
    return sim_val, picked


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
        densities = remaining_values[idxs] / remaining_weights[idxs]
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
    densities = f_v / f_w
    order = np.argsort(-densities)  # local indices, best density first

    # 1-step lookahead: for each candidate top item, simulate the rest
    # and pick the candidate whose resulting total value is highest.
    # Limit candidates to top K by density to keep cost low.
    K = min(60, m)
    candidates = order[:K]

    # Precompute the full order list of local indices
    full_order = order.tolist()

    # Phase 1: Fast screening with simple greedy simulation
    best_val = -1.0
    best_local = -1
    candidate_scores = []
    for cand in candidates:
        val = _simulate_greedy(remaining_capacity, f_w, f_v, full_order, cand)
        candidate_scores.append((val, cand))
        if val > best_val:
            best_val = val
            best_local = cand
        if not _elapsed_ok():
            break

    # Phase 2: Accurate simulation for top candidates
    if not _elapsed_ok():
        if best_local < 0:
            best_local = int(candidates[0])
        return int(feasible_idx[best_local])

    # Sort by fast score, take top L for accurate simulation
    candidate_scores.sort(key=lambda x: -x[0])
    L = min(15, len(candidate_scores))
    top_candidates = [c for _, c in candidate_scores[:L]]

    best_accurate_val = -1.0
    best_accurate_local = best_local
    best_accurate_picked = None
    
    for cand in top_candidates:
        val, picked = _simulate_greedy_accurate(remaining_capacity, f_w, f_v, full_order, cand)
        if val > best_accurate_val:
            best_accurate_val = val
            best_accurate_local = cand
            best_accurate_picked = picked
        if not _elapsed_ok():
            break

    # Phase 3: Local swap polish on the best solution found
    if best_accurate_picked is not None and _elapsed_ok():
        picked_set = set(best_accurate_picked)
        sim_cap = remaining_capacity
        for idx in best_accurate_picked:
            sim_cap -= f_w[idx]
        
        # Try swapping out each picked item for an unpicked feasible item
        improved = True
        while improved and _elapsed_ok():
            improved = False
            for out_idx in list(picked_set):
                w_out = f_w[out_idx]
                v_out = f_v[out_idx]
                temp_cap = sim_cap + w_out
                # Find best item to swap in
                for in_idx in range(m):
                    if in_idx in picked_set:
                        continue
                    w_in = f_w[in_idx]
                    if w_in <= temp_cap:
                        v_in = f_v[in_idx]
                        if v_in > v_out:
                            # Perform swap
                            picked_set.remove(out_idx)
                            picked_set.add(in_idx)
                            sim_cap = temp_cap - w_in
                            improved = True
                            break
                if improved:
                    break

    if best_local < 0:
        best_local = int(candidates[0])

    # Map local index back to index in remaining arrays
    return int(feasible_idx[best_accurate_local if best_accurate_local >= 0 else best_local])
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s kill


def _greedy_complete(capacity, weights, values):
    """Complete greedily by value density; return total value."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    total = 0.0
    cap = capacity
    # Pre-sort by density for faster repeated selection
    densities = values / weights
    order = np.argsort(-densities)
    # Track which items are still available
    available = np.ones(n, dtype=bool)
    
    for _ in range(n):
        if cap <= 0:
            break
        # Find best feasible item from pre-sorted order
        found = False
        for i in order:
            if not available[i]:
                continue
            if weights[i] <= cap:
                total += values[i]
                cap -= weights[i]
                available[i] = False
                found = True
                break
        if not found:
            break
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Pick the next item using 1-step lookahead greedy with swap awareness."""
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Filter to feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        return -1

    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Compute densities for feasible items
    fw = remaining_weights[feasible_indices]
    fv = remaining_values[feasible_indices]
    densities = fv / fw

    # Sort by density descending, take top candidates for look-ahead
    # Use a composite score: density + potential future value
    # Items with moderate density but leaving room for many good items score well
    max_candidates = min(len(feasible_indices), 20)
    order = np.argsort(-densities)
    
    # Score each candidate: immediate value + estimated future value from remaining capacity
    # Future value estimate: sum of top densities that fit in remaining capacity
    candidate_scores = np.zeros(len(order))
    for rank, ci in enumerate(order):
        picked_w = fw[ci]
        picked_v = fv[ci]
        rem_cap = remaining_capacity - picked_w
        # Estimate: pick this, then greedily fill rest
        # Quick estimate: use top items by density that fit
        # For speed, just use picked_v + rem_cap * avg_density of remaining
        if rank < len(order) - 1:
            other_densities = np.delete(densities[order], rank)
            # Average density of remaining feasible items
            avg_d = np.mean(other_densities) if len(other_densities) > 0 else 0
        else:
            avg_d = 0
        candidate_scores[rank] = picked_v + rem_cap * avg_d
    
    # Take top candidates by this composite score
    top_order = np.argsort(-candidate_scores)
    top_candidates = top_order[:max_candidates]

    best_simulated_total = -1.0
    best_candidate_idx = int(feasible_indices[top_candidates[0]])

    for ci in top_candidates:
        # Simulate: pick this item, then greedily fill the rest
        picked_weight = fw[ci]
        picked_value = fv[ci]
        remaining_cap = remaining_capacity - picked_weight

        # Build arrays without this item
        other_mask = np.ones(len(feasible_indices), dtype=bool)
        other_mask[ci] = False
        other_idx = feasible_indices[other_mask]
        other_w = remaining_weights[other_idx]
        other_v = remaining_values[other_idx]

        sim_total = picked_value + _greedy_complete(remaining_cap, other_w, other_v)

        if sim_total > best_simulated_total:
            best_simulated_total = sim_total
            best_candidate_idx = int(feasible_indices[ci])

    # Deadline check before returning
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -1

    # 2-step lookahead refinement for the top 3 candidates
    if len(top_candidates) >= 2 and time.time() - _START_TIME < _TIME_LIMIT - 5:
        top3 = top_candidates[:3]
        best_2step_total = -1.0
        best_2step_idx = best_candidate_idx
        
        for ci1 in top3:
            w1 = fw[ci1]
            v1 = fv[ci1]
            cap1 = remaining_capacity - w1
            if cap1 <= 0:
                continue
            
            # For second pick, consider top candidates by density from remaining
            mask1 = np.ones(len(feasible_indices), dtype=bool)
            mask1[ci1] = False
            idx1 = feasible_indices[mask1]
            w1_arr = remaining_weights[idx1]
            v1_arr = remaining_values[idx1]
            
            # Filter feasible for second pick
            feas2 = w1_arr <= cap1
            if not np.any(feas2):
                # Only one step
                total_2 = v1 + _greedy_complete(cap1, w1_arr[~feas2] if np.any(~feas2) else np.array([]), v1_arr[~feas2] if np.any(~feas2) else np.array([]))
                if total_2 > best_2step_total:
                    best_2step_total = total_2
                    best_2step_idx = int(feasible_indices[ci1])
                continue
            
            feas2_idx = np.where(feas2)[0]
            # Take top 5 by density for second pick
            d2 = v1_arr[feas2_idx] / w1_arr[feas2_idx]
            top5 = feas2_idx[np.argsort(-d2)][:5]
            
            for ci2 in top5:
                w2 = w1_arr[ci2]
                v2 = v1_arr[ci2]
                cap2 = cap1 - w2
                if cap2 < 0:
                    continue
                
                # Greedy complete for the rest
                mask2 = np.ones(len(idx1), dtype=bool)
                mask2[ci2] = False
                idx2 = idx1[mask2]
                w2_arr = remaining_weights[idx2]
                v2_arr = remaining_values[idx2]
                
                total_2 = v1 + v2 + _greedy_complete(cap2, w2_arr, v2_arr)
                
                if total_2 > best_2step_total:
                    best_2step_total = total_2
                    best_2step_idx = int(feasible_indices[ci1])  # First pick determines the answer
        
        if best_2step_total > best_simulated_total:
            best_candidate_idx = best_2step_idx

    return best_candidate_idx
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_DEADLINE = None


def _get_deadline():
    """Get or set the global deadline (150s wall-clock with safety margin)."""
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 145.0  # 5s safety margin
    return _GLOBAL_DEADLINE


def _greedy_fill_vectorized(remaining_capacity, remaining_weights, remaining_values, order):
    """
    Greedily fill the knapsack by value density using precomputed order.
    Returns the total value achievable by this greedy strategy.
    """
    if remaining_capacity <= 0 or len(remaining_weights) == 0:
        return 0.0
    
    total_value = 0.0
    cap = remaining_capacity
    for idx in order:
        w = remaining_weights[idx]
        if w <= cap:
            total_value += remaining_values[idx]
            cap -= w
        if cap <= 0:
            break
    
    return total_value


def _one_step_lookahead_fast(remaining_capacity, remaining_weights, remaining_values, deadline):
    """
    Vectorized 1-step lookahead: for each feasible item, compute total value
    if picked first, then greedily fill the rest.
    Uses precomputed density order and efficient simulation.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    n_feasible = len(feasible_indices)
    
    if n_feasible == 1:
        return int(feasible_indices[0])
    
    # Precompute densities and sort order
    densities = remaining_values / remaining_weights
    all_order = np.argsort(-densities, kind='stable')
    
    # For each candidate, we need to simulate greedy fill excluding that item
    # Key optimization: precompute prefix sums of values and weights in density order
    # But we need to skip items that don't fit, which makes it tricky
    
    # Alternative: for each candidate, do a fast loop
    # Since n is at most 100, and we have deadline checks, let's do it efficiently
    
    best_score = -1.0
    best_idx = -1
    
    # Pre-filter: only consider feasible items as candidates
    # For each candidate, simulate greedy on the rest
    
    # Optimization: precompute the sorted order of feasible items only
    feasible_densities = densities[feasible_indices]
    feasible_order_local = np.argsort(-feasible_densities, kind='stable')
    feasible_order_global = feasible_indices[feasible_order_local]
    
    # For each candidate in feasible_order_global, compute total value
    for cand_pos in range(n_feasible):
        if time.time() > deadline:
            break
        
        cand_idx = feasible_order_global[cand_pos]
        cand_w = remaining_weights[cand_idx]
        cand_v = remaining_values[cand_idx]
        
        cap_after = remaining_capacity - cand_w
        total_v = cand_v
        
        # Greedy fill: iterate through all_order, skip cand_idx
        for item_idx in all_order:
            if item_idx == cand_idx:
                continue
            w = remaining_weights[item_idx]
            if w <= cap_after:
                total_v += remaining_values[item_idx]
                cap_after -= w
            if cap_after <= 0:
                break
        
        if total_v > best_score:
            best_score = total_v
            best_idx = int(cand_idx)
    
    if best_idx != -1:
        return best_idx
    
    # Fallback: pure density
    return int(feasible_indices[np.argmax(densities[feasible_indices])])


def _two_step_lookahead(remaining_capacity, remaining_weights, remaining_values, deadline, top_k=5):
    """
    2-step lookahead: try each of the top-k candidates by 1-step score,
    then for each, try the best 2nd candidate with greedy completion.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    n_feasible = len(feasible_indices)
    
    if n_feasible == 1:
        return int(feasible_indices[0])
    
    densities = remaining_values / remaining_weights
    all_order = np.argsort(-densities, kind='stable')
    
    # First, compute 1-step scores for all feasible candidates
    one_step_scores = {}
    for cand_idx in feasible_indices:
        if time.time() > deadline:
            break
        cand_w = remaining_weights[cand_idx]
        cand_v = remaining_values[cand_idx]
        cap_after = remaining_capacity - cand_w
        total_v = cand_v
        
        for item_idx in all_order:
            if item_idx == cand_idx:
                continue
            w = remaining_weights[item_idx]
            if w <= cap_after:
                total_v += remaining_values[item_idx]
                cap_after -= w
            if cap_after <= 0:
                break
        
        one_step_scores[cand_idx] = total_v
    
    if not one_step_scores:
        return int(feasible_indices[np.argmax(densities[feasible_indices])])
    
    # Get top-k candidates by 1-step score
    sorted_candidates = sorted(one_step_scores.items(), key=lambda x: -x[1])[:top_k]
    
    # For each top candidate, try 2-step lookahead
    best_total = -1.0
    best_idx = -1
    
    for cand1_idx, score1 in sorted_candidates:
        if time.time() > deadline:
            break
        
        cand1_w = remaining_weights[cand1_idx]
        cand1_v = remaining_values[cand1_idx]
        cap_after1 = remaining_capacity - cand1_w
        
        # Try each other feasible item as second pick
        for cand2_idx in feasible_indices:
            if cand2_idx == cand1_idx:
                continue
            if remaining_weights[cand2_idx] > cap_after1:
                continue
            
            cand2_w = remaining_weights[cand2_idx]
            cand2_v = remaining_values[cand2_idx]
            cap_after2 = cap_after1 - cand2_w
            total_v = cand1_v + cand2_v
            
            # Greedy fill with remaining
            for item_idx in all_order:
                if item_idx == cand1_idx or item_idx == cand2_idx:
                    continue
                w = remaining_weights[item_idx]
                if w <= cap_after2:
                    total_v += remaining_values[item_idx]
                    cap_after2 -= w
                if cap_after2 <= 0:
                    break
            
            if total_v > best_total:
                best_total = total_v
                best_idx = int(cand1_idx)  # Return first pick
    
    if best_idx != -1:
        return best_idx
    
    # Fallback: best 1-step candidate
    return int(sorted_candidates[0][0])


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Returns an index into the remaining (feasible) arrays, or -1 to stop.
    """
    if len(remaining_weights) == 0 or remaining_capacity <= 0:
        return -1
    
    deadline = _get_deadline()
    
    # Check if any item fits
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    n_feasible = np.sum(feasible_mask)
    
    # Use 2-step lookahead for small n, 1-step for larger
    if n_feasible <= 30 and time.time() < deadline - 0.1:
        return _two_step_lookahead(remaining_capacity, remaining_weights, remaining_values, deadline, top_k=3)
    else:
        return _one_step_lookahead_fast(remaining_capacity, remaining_weights, remaining_values, deadline)
# EVOLVE-BLOCK-END

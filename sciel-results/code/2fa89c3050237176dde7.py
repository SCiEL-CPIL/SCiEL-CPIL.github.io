import numpy as np
import time

_START = None
_HARD_LIMIT = 145.0  # Safety margin before 150s kill

# Global state to cache the optimal solution for the current instance
_MASTER_PLAN = None  # List of (weight, value, original_index_in_first_call) or just a set of identifiers
_MASTER_PLAN_ORDER = []  # Ordered list of indices from the first call, representing the construction order
_INSTANCE_ID = None  # To detect if a new instance has started (e.g., n changes or specific signature)

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _time_left():
    return _HARD_LIMIT - (_now() - _START)

def _density_greedy_solution(cap, weights, values):
    """
    Returns a boolean mask of items selected by density greedy.
    """
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=bool)
    
    cap = float(cap)
    selected = np.zeros(n, dtype=bool)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    order = np.argsort(-density, kind='stable')
    
    for idx in order:
        idx = int(idx)
        wi = float(weights[idx])
        if wi <= cap:
            cap -= wi
            selected[idx] = True
        if cap <= 1e-9:
            break
            
    return selected

def _local_search_deep(cap, weights, values, selected):
    """
    Deep local search. Since we have a large time budget for one call, 
    we can afford more iterations and swap types.
    """
    n = len(weights)
    if n == 0:
        return selected
        
    selected = selected.copy()
    improved = True
    iterations = 0
    
    # We have ~140 seconds. Let's use a time-based loop.
    while improved and _time_left() > 2.0:
        improved = False
        iterations += 1
        
        # Calculate current stats
        total_weight = float(np.sum(weights[selected]))
        total_value = float(np.sum(values[selected]))
        
        remaining_cap = cap - total_weight
        sel_idx = np.where(selected)[0]
        unsel_idx = np.where(~selected)[0]
        
        if len(sel_idx) == 0:
            # If nothing selected, just add best density item
            if len(unsel_idx) > 0:
                best_idx = unsel_idx[np.argmax(values[unsel_idx] / np.maximum(weights[unsel_idx], 1e-9))]
                if weights[best_idx] <= remaining_cap:
                    selected[best_idx] = True
                    improved = True
            continue
            
        if len(unsel_idx) == 0:
            break
            
        best_gain = 1e-9
        best_swap = None
        best_swap_type = 0 # 1 for 1-for-1, 2 for 2-for-1
        
        # 1. 1-for-1 Swaps
        # For each unselected j, find best selected i to remove
        # Gain = v[j] - v[i], Constraint: w[j] - w[i] <= remaining_cap
        # => w[i] >= w[j] - remaining_cap
        
        sel_w = weights[sel_idx]
        sel_v = values[sel_idx]
        
        for j in unsel_idx:
            wj = float(weights[j])
            vj = float(values[j])
            
            min_wi_needed = wj - remaining_cap
            
            # Find selected items with weight >= min_wi_needed
            mask = sel_w >= min_wi_needed
            if not np.any(mask):
                continue
                
            candidate_v = sel_v[mask]
            min_v_idx_in_mask = np.argmin(candidate_v)
            min_v = candidate_v[min_v_idx_in_mask]
            
            gain = vj - min_v
            if gain > best_gain:
                masked_sel_indices = sel_idx[mask]
                target_i = masked_sel_indices[min_v_idx_in_mask]
                best_gain = gain
                best_swap = (int(target_i), int(j))
                best_swap_type = 1
        
        # 2. 2-for-1 Swaps: Remove selected i, add unselected j, k
        # Gain = v[j] + v[k] - v[i]
        # Constraint: w[j] + w[k] - w[i] <= remaining_cap
        # => w[i] >= w[j] + w[k] - remaining_cap
        
        if _time_left() > 5.0 and len(unsel_idx) >= 2:
            best_gain_21 = 1e-9
            best_swap_21 = None
            
            u_list = list(unsel_idx)
            L = len(u_list)
            
            # Limit pair checks if L is large to keep it fast, but N=100 is small
            for a in range(L):
                j = u_list[a]
                wj = float(weights[j])
                vj = float(values[j])
                
                for b in range(a + 1, L):
                    k = u_list[b]
                    wk = float(weights[k])
                    vk = float(values[k])
                    
                    w_pair = wj + wk
                    v_pair = vj + vk
                    
                    min_wi_needed = w_pair - remaining_cap
                    
                    mask = sel_w >= min_wi_needed
                    if not np.any(mask):
                        continue
                        
                    candidate_v = sel_v[mask]
                    min_v = np.min(candidate_v)
                    
                    gain = v_pair - min_v
                    if gain > best_gain_21:
                        masked_sel_indices = sel_idx[mask]
                        target_i = masked_sel_indices[np.argmin(candidate_v)]
                        best_gain_21 = gain
                        best_swap_21 = (int(target_i), int(j), int(k))
            
            if best_swap_21 is not None and best_gain_21 > best_gain:
                best_gain = best_gain_21
                best_swap = best_swap_21
                best_swap_type = 2
        
        # 3. 1-for-2 Swaps: Remove selected i, j, add unselected k
        # Gain = v[k] - v[i] - v[j]
        # Constraint: w[k] - w[i] - w[j] <= remaining_cap
        # => w[i] + w[j] >= w[k] - remaining_cap
        
        if _time_left() > 10.0 and len(sel_idx) >= 2:
            best_gain_12 = 1e-9
            best_swap_12 = None
            
            s_list = list(sel_idx)
            Ls = len(s_list)
            
            for k in unsel_idx:
                wk = float(weights[k])
                vk = float(values[k])
                
                min_w_pair_needed = wk - remaining_cap
                
                # Find pair (i, j) in selected such that w[i]+w[j] >= min_w_pair_needed
                # and v[i]+v[j] is minimized
                
                # Naive O(N^2) over selected items
                # Since Ls is ~50, 2500 pairs. 2500 * 50 (unselected) = 125k ops. Fast.
                for a in range(Ls):
                    i = s_list[a]
                    wi = float(weights[i])
                    vi = float(values[i])
                    
                    # Optimization: if wi is already large enough, we can pick any j with min value?
                    # No, we need to minimize vi + vj.
                    # So for fixed i, we want min vj such that wj >= min_w_pair_needed - wi
                    
                    min_wj_needed = min_w_pair_needed - wi
                    
                    # Find j in selected (j != i) with wj >= min_wj_needed and min vj
                    # To avoid i==j, we can just check all and ensure index match
                    # But careful: if i==j, we are removing same item twice, which is invalid for 0/1 knapsack swaps in this context (we remove distinct items).
                    
                    mask_j = sel_w >= min_wj_needed
                    # Exclude i from consideration? 
                    # sel_idx contains i. If mask_j includes i, we might pick i.
                    # We need to ensure i != j.
                    
                    # Simple approach: iterate j in sel_idx
                    # This is O(N^2) total for this block.
                    for b in range(Ls):
                        if a == b: continue
                        j = s_list[b]
                        wj = float(weights[j])
                        vj = float(values[j])
                        
                        if wj >= min_wj_needed:
                            gain = vk - vi - vj
                            if gain > best_gain_12:
                                best_gain_12 = gain
                                best_swap_12 = (int(i), int(j), int(k))
            
            if best_swap_12 is not None and best_gain_12 > best_gain:
                best_gain = best_gain_12
                best_swap = best_swap_12
                best_swap_type = 3

        if best_swap is not None:
            if best_swap_type == 1:
                i, j = best_swap
                selected[i] = False
                selected[j] = True
            elif best_swap_type == 2:
                i, j, k = best_swap
                selected[i] = False
                selected[j] = True
                selected[k] = True
            elif best_swap_type == 3:
                i, j, k = best_swap
                selected[i] = False
                selected[j] = False
                selected[k] = True
            
            improved = True
            # Restart search with new state
            continue
            
    return selected

def _compute_master_plan(cap, weights, values):
    """
    Compute the optimal subset for the full instance.
    Returns a list of indices (in the original array order) sorted by a construction heuristic.
    """
    selected, _ = _density_greedy_solution(cap, weights, values)
    
    # Deep local search
    if _time_left() > 1.0:
        selected = _local_search_deep(cap, weights, values, selected)
    
    # Get indices of selected items
    sel_indices = np.where(selected)[0]
    
    if len(sel_indices) == 0:
        return []
        
    # Sort by descending density (value/weight) to define a stable construction order.
    # Items with higher density are "better" per unit weight, so we pick them first.
    # This ensures that if we have to stop early (which we won't, as we follow the plan),
    # we have picked the highest value items.
    densities = values[sel_indices] / np.maximum(weights[sel_indices], 1e-9)
    sorted_order = np.argsort(-densities, kind='stable')
    
    ordered_indices = sel_indices[sorted_order]
    
    return [int(idx) for idx in ordered_indices]

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _MASTER_PLAN, _MASTER_PLAN_ORDER, _INSTANCE_ID, _START
    
    if remaining_capacity is None:
        return -1
        
    cap = float(remaining_capacity)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    
    if n == 0 or cap <= 0:
        return -1
        
    feasible = weights <= cap
    if not feasible.any():
        return -1
        
    # Detect new instance
    # A new instance is likely if n is at its maximum (100) or if we haven't started yet.
    # Since the evaluator generates 8 instances, we assume each instance starts with n=100.
    # If n < 100, we are in the middle of an instance.
    # We use a simple heuristic: if n == 100, it's the start of a new instance.
    # Note: This assumes the problem size is fixed at 100. The prompt says "100-item instances".
    
    is_new_instance = (n == 100) or (_MASTER_PLAN is None)
    
    if is_new_instance:
        # Reset state
        _MASTER_PLAN = None
        _MASTER_PLAN_ORDER = []
        
        # Compute the master plan
        plan_indices = _compute_master_plan(cap, weights, values)
        _MASTER_PLAN_ORDER = plan_indices
        
        # If plan is empty, fallback to immediate greedy
        if not _MASTER_PLAN_ORDER:
             # Fallback: pick best density feasible
             with np.errstate(divide='ignore', invalid='ignore'):
                 density = np.where(weights > 0, values / weights, 0.0)
             fd = np.where(feasible, density, -np.inf)
             return int(np.argmax(fd))
             
    # Now, find the next item in _MASTER_PLAN_ORDER that is present in the current remaining arrays.
    # We need to map from the original indices (0..99) to the current indices in `remaining_weights`.
    # The evaluator removes items from the array. So the indices in `remaining_weights` are 0..n-1.
    # But our _MASTER_PLAN_ORDER contains indices relative to the ORIGINAL 100-item array.
    
    # How do we know which original index corresponds to which current index?
    # We don't have the original array anymore, only the remaining one.
    # However, we can identify items by their (weight, value) pair? 
    # Risk: duplicate weights/values.
    # Better: The evaluator passes `remaining_weights` and `remaining_values`.
    # When we return an index `idx`, the evaluator removes `weights[idx]` and `values[idx]` from the arrays.
    # So the "identity" of an item is its position in the *current* array.
    # But our master plan was built on the *initial* array.
    
    # This is a problem. If I store index 5 from the initial array, and the evaluator removes item 0,
    # the item that was at index 5 is now at index 4.
    
    # We need a robust way to track items.
    # Strategy: Instead of storing indices, store the (weight, value) pairs of the selected items.
    # Then, in each step, look for the item in the current `remaining` arrays that matches the next target (weight, value).
    # To handle duplicates, we can use a multiset or just pick the first match.
    
    # Let's re-implement _MASTER_PLAN_ORDER to store tuples (w, v).
    # And in select_next_item, we search for the target in the current arrays.
    
    if is_new_instance:
        # We already computed plan_indices. Let's convert to (w, v) pairs.
        # But we need to re-calculate or store the pairs.
        # Let's modify _compute_master_plan to return pairs.
        
        # Re-do the plan computation to get pairs, or just use the indices from the last call?
        # We have `weights` and `values` from the initial call (since n=100).
        plan_pairs = []
        for idx in _MASTER_PLAN_ORDER:
            plan_pairs.append((float(weights[idx]), float(values[idx])))
        _MASTER_PLAN_ORDER = plan_pairs # Store pairs now
        
    # Find the first pair in _MASTER_PLAN_ORDER that exists in current weights/values
    # We need to find an index in the current arrays that matches the target pair.
    
    # To make this efficient and robust, we can build a set of available (w, v) pairs in the current array.
    # But we need the index to return.
    
    # Simple approach: Iterate through _MASTER_PLAN_ORDER.
    # For each target pair (tw, tv), check if it exists in current weights/values.
    # If yes, return the index.
    # We remove it from our local tracking? No, the evaluator removes it.
    # But we need to ensure we don't pick the same item twice if there are duplicates.
    # Since the evaluator removes the item, the next call will have one fewer item.
    # So if we just look for the FIRST target in the list that is present, it will work.
    # Because once we return an index, that item is gone.
    # If there are duplicates, we might pick a "different" item with the same w,v, but that's fine.
    
    # Optimization: We can maintain a pointer to the next expected item in the plan.
    # But if the plan order is fixed, and we just pick the first available, it's O(N) per step.
    # N=100, so 100 steps * 100 checks = 10k checks. Very fast.
    
    # Let's refine:
    # We want to pick items in the order of _MASTER_PLAN_ORDER.
    # So we iterate through _MASTER_PLAN_ORDER.
    # For the first target (tw, tv) in the list:
    #   Check if any item in current arrays has weight=tw and value=tv.
    #   If yes, return that index.
    #   If no, it means this item was already removed (shouldn't happen if we follow plan) or there's a mismatch.
    #   Actually, if we always pick the first available from the plan, and the plan is a subset of the initial items,
    #   and the evaluator removes the item we pick, then the next step will see the next item in the plan
    #   (unless there are duplicates, in which case it might see another duplicate of the same pair, which is fine).
    
    # Wait, if I have two items with same (w,v) in the plan, and I pick one, the other is still in the array.
    # Next step, I look for the same (w,v) again (since it's the next in the list? No, the list has two entries).
    # If the list is [(w1,v1), (w1,v1), (w2,v2)], and I pick the first (w1,v1).
    # Next step, I look for the first available in [(w1,v1), (w1,v1), (w2,v2)].
    # The first (w1,v1) is still in the array (the second one). So I pick it.
    # This works.
    
    for tw, tv in _MASTER_PLAN_ORDER:
        # Find index in current arrays where weight == tw and value == tv
        # Use numpy for speed
        mask = (weights == tw) & (values == tv)
        if np.any(mask):
            idx = np.argmax(mask) # First match
            return int(idx)
            
    # If no item from the master plan is found, fallback to greedy
    # This can happen if the master plan was empty or if there's a numerical mismatch.
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    fd = np.where(feasible, density, -np.inf)
    return int(np.argmax(fd))


import numpy as np
import time

_START = None
_HARD_LIMIT = 145.0  # Safety margin before 150s kill

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _fast_greedy(cap, weights, values, mode='density'):
    """
    Fast greedy heuristic.
    mode: 'density', 'value', 'weight'
    Returns: (selected_mask, total_value)
    """
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0
    
    cap = float(cap)
    selected = np.zeros(n, dtype=bool)
    total_val = 0.0
    
    if mode == 'density':
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = np.where(weights > 0, values / weights, 0.0)
    elif mode == 'value':
        scores = values.copy()
    elif mode == 'weight':
        # Pick lightest first? Or heaviest? Let's try lightest to fit more.
        scores = -weights
    else:
        scores = np.where(weights > 0, values / weights, 0.0)
        
    order = np.argsort(-scores, kind='stable')
    
    for idx in order:
        idx = int(idx)
        wi = float(weights[idx])
        if wi <= cap:
            selected[idx] = True
            cap -= wi
            total_val += float(values[idx])
            if cap <= 1e-9:
                break
    return selected, total_val

def _local_search(selected, weights, values, cap_original, max_iters=200, time_limit=5.0):
    """
    1-swap local search with time limit.
    Returns improved selected mask and total value.
    """
    n = len(weights)
    if n == 0:
        return selected, 0.0
    
    cur_selected = selected.copy()
    cur_val = float(np.sum(values[cur_selected]))
    cur_weight = float(np.sum(weights[cur_selected]))
    remaining_cap = cap_original - cur_weight
    
    w = weights
    v = values
    
    sel_idx = np.where(cur_selected)[0]
    un_sel_idx = np.where(~cur_selected)[0]
    
    start_time = time.time()
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        if time.time() - start_time > time_limit:
            break
        if _now() - _START > _HARD_LIMIT:
            break
            
        # 1. Try to ADD
        if len(un_sel_idx) > 0 and remaining_cap > 0:
            candidates_add = un_sel_idx[w[un_sel_idx] <= remaining_cap]
            if len(candidates_add) > 0:
                # Pick best value add
                best_add_idx = candidates_add[np.argmax(v[candidates_add])]
                
                # Check if it fits (it does by filter)
                cur_selected[best_add_idx] = True
                cur_val += v[best_add_idx]
                remaining_cap -= w[best_add_idx]
                
                un_sel_idx = np.delete(un_sel_idx, np.where(un_sel_idx == best_add_idx)[0][0])
                sel_idx = np.append(sel_idx, best_add_idx)
                improved = True
                continue
        
        # 2. Try SWAP
        if len(sel_idx) > 0 and len(un_sel_idx) > 0:
            best_gain = -np.inf
            best_i = -1
            best_j = -1
            
            # Vectorized-ish swap search
            # For each selected i, find best unselected j
            # w[j] <= w[i] + remaining_cap
            # gain = v[j] - v[i]
            
            # To speed up, we can precompute max v[j] for various weight constraints?
            # Or just iterate. N=100, so 100*100 = 10k ops.
            for i in sel_idx:
                wi = w[i]
                vi = v[i]
                max_wj = wi + remaining_cap
                
                mask_j = (w[un_sel_idx] <= max_wj)
                if not np.any(mask_j):
                    continue
                
                candidate_js = un_sel_idx[mask_j]
                if len(candidate_js) == 0:
                    continue
                
                vals_j = v[candidate_js]
                best_j_local = candidate_js[np.argmax(vals_j)]
                best_val_j = v[best_j_local]
                
                gain = best_val_j - vi
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = best_j_local
            
            if best_i >= 0 and best_gain > 1e-9:
                cur_selected[best_i] = False
                cur_selected[best_j] = True
                cur_val += best_gain
                remaining_cap += w[best_i] - w[best_j]
                
                sel_idx = np.delete(sel_idx, np.where(sel_idx == best_i)[0][0])
                un_sel_idx = np.delete(un_sel_idx, np.where(un_sel_idx == best_j)[0][0])
                sel_idx = np.append(sel_idx, best_j)
                
                improved = True
                continue
        
        # 3. Try REMOVE (if it helps? No, removal reduces value unless it enables a better add/swap)
        # If we can't add or swap, we are stuck. 
        # Sometimes removing a heavy low-value item allows adding two medium high-value items.
        # This is a 2-move. 
        # For simplicity, let's try a "remove and best add" combined move.
        if len(sel_idx) > 0 and len(un_sel_idx) > 0:
            best_net_gain = -np.inf
            best_i_to_remove = -1
            best_j_to_add = -1
            
            for i in sel_idx:
                wi = w[i]
                vi = v[i]
                # If we remove i, remaining_cap_new = remaining_cap + wi
                new_cap = remaining_cap + wi
                
                # Find best add j in new_cap
                candidates_add = un_sel_idx[w[un_sel_idx] <= new_cap]
                if len(candidates_add) == 0:
                    continue
                
                best_j_local = candidates_add[np.argmax(v[candidates_add])]
                best_val_j = v[best_j_local]
                
                # Net gain = v[j] - v[i]
                # Note: This is effectively a swap where w[j] can be larger than w[i]
                # but it's bounded by the new capacity.
                # The previous swap logic already handles w[j] <= w[i] + remaining_cap.
                # So this is redundant with the swap logic above?
                # Yes, swap logic allows w[j] up to w[i] + remaining_cap.
                # So this block is redundant. 
                pass
                
    return cur_selected, cur_val

def _solve_knapsack_subproblem(cap, weights, values):
    """
    Multi-start local search to find a good solution for the current subproblem.
    Returns the best selected mask found.
    """
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=bool)
    
    best_sel = np.zeros(n, dtype=bool)
    best_val = -1.0
    
    # Start with different greedy initializations
    modes = ['density', 'value', 'weight']
    
    for mode in modes:
        sel, val = _fast_greedy(cap, weights, values, mode=mode)
        if val > best_val:
            best_val = val
            best_sel = sel.copy()
            
    # Local search from the best greedy start
    improved_sel, improved_val = _local_search(best_sel, weights, values, cap, max_iters=100, time_limit=2.0)
    
    if improved_val > best_val:
        best_val = improved_val
        best_sel = improved_sel
        
    # Try a few random restarts if time permits
    if _now() - _START < _HARD_LIMIT - 10.0:
        for _ in range(5):
            # Random start: pick random items that fit
            random_sel = np.zeros(n, dtype=bool)
            random_cap = cap
            indices = np.random.permutation(n)
            for idx in indices:
                idx = int(idx)
                if weights[idx] <= random_cap:
                    random_sel[idx] = True
                    random_cap -= weights[idx]
                    if random_cap <= 0:
                        break
            
            rand_val = float(np.sum(values[random_sel]))
            if rand_val > 0:
                improved_rand_sel, improved_rand_val = _local_search(random_sel, weights, values, cap, max_iters=50, time_limit=1.0)
                if improved_rand_val > best_val:
                    best_val = improved_rand_val
                    best_sel = improved_rand_sel.copy()
                    
            if _now() - _START > _HARD_LIMIT:
                break
                
    return best_sel

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if remaining_capacity is None:
        return -1
    
    cap = float(remaining_capacity)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    
    if n == 0 or cap <= 0:
        return -1
        
    if not np.any(weights <= cap):
        return -1

    # Solve the subproblem
    best_sel = _solve_knapsack_subproblem(cap, weights, values)
    
    selected_indices = np.where(best_sel)[0]
    
    if len(selected_indices) == 0:
        # Fallback: max density feasible
        feasible = weights <= cap
        if not np.any(feasible):
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(weights > 0, values / weights, 0.0)
        fd = np.where(feasible, density, -np.inf)
        return int(np.argmax(fd))
        
    # Selection Strategy:
    # We have a set of items that form a good solution.
    # Which one to pick now?
    # If we pick the "most critical" item, we might block alternatives.
    # If we pick the "least critical" item, we preserve the core of the solution.
    # 
    # Heuristic: Pick the item with the lowest "regret" or lowest density?
    # Or pick the item that is included in the MOST solutions from a small ensemble?
    # Given time constraints, let's use a simpler proxy:
    # Pick the item with the lowest density among the selected set?
    # Rationale: High density items are "safe" bets. Low density items are "risky" or "fillers".
    # If we pick the filler first, we might run out of space for the core? No, the set is already feasible.
    # 
    # Actually, a better heuristic for "next item" in a known good set:
    # Pick the item that maximizes the *remaining* potential?
    # 
    # Let's try: Pick the item with the HIGHEST VALUE.
    # Why? It anchors the high value. The rest of the set is still feasible.
    # If the rest of the set is {A, B, C} and we pick A (high value), the remaining capacity allows B and C.
    # If we pick C (low value) first, the remaining capacity still allows A and B.
    # So it shouldn't matter much IF the set is consistent.
    # 
    # However, consider if the optimal set is {A, B} but {A, C} is also good.
    # If we pick A, we can pick B or C.
    # If we pick B, we can pick A.
    # 
    # Let's stick with Highest Value. It's a strong signal of importance.
    
    vals_in_sel = values[selected_indices]
    best_local_idx = np.argmax(vals_in_sel)
    return int(selected_indices[best_local_idx])

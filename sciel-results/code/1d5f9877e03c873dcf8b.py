import numpy as np
import time

_START = None
_HARD_LIMIT = 140.0  # Safety margin before 150s kill

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _density_greedy_solution(cap, weights, values):
    """
    Returns a boolean mask of items selected by density greedy,
    and the total value.
    """
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0
    
    cap = float(cap)
    selected = np.zeros(n, dtype=bool)
    total_val = 0.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    
    # Sort indices by density descending
    order = np.argsort(-density, kind='stable')
    
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

def _improve_solution(selected, weights, values, cap_original):
    """
    Perform local search (hill climbing) on the boolean selection mask.
    Moves:
    1. Add an unselected item if it fits.
    2. Remove a selected item if it frees up space for a better combination (handled via swap).
    3. Swap a selected item with an unselected item if it improves value and fits.
    
    Returns the improved selected mask and total value.
    """
    n = len(weights)
    if n == 0:
        return selected, 0.0
    
    # Current state
    cur_selected = selected.copy()
    cur_val = float(np.sum(values[cur_selected]))
    cur_weight = float(np.sum(weights[cur_selected]))
    remaining_cap = cap_original - cur_weight
    
    # Precompute arrays for speed
    w = weights
    v = values
    
    # Indices of selected and unselected
    sel_idx = np.where(cur_selected)[0]
    un_sel_idx = np.where(~cur_selected)[0]
    
    improved = True
    max_iters = 50 # Limit iterations to prevent infinite loops or excessive time
    
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        
        # Check time
        if _now() - _START > _HARD_LIMIT:
            break
            
        # 1. Try to ADD any unselected item that fits
        # Filter unselected items that fit in remaining capacity
        if len(un_sel_idx) > 0 and remaining_cap > 0:
            # Find unselected items that fit
            # We can just iterate, but numpy is faster for bulk check
            candidates_add = un_sel_idx[w[un_sel_idx] <= remaining_cap]
            if len(candidates_add) > 0:
                # Pick the one with highest value (or density? Let's try value first as it's direct)
                # Actually, picking highest value that fits is a good greedy move
                best_add_idx = candidates_add[np.argmax(v[candidates_add])]
                
                # Check if adding it is actually an improvement (it should be, since value > 0 usually)
                # But we must ensure it doesn't block better future items? 
                # Local search assumes current state is fixed. Adding a positive value item is always an improvement 
                # IF it doesn't prevent adding a much larger item later. 
                # However, simple hill climbing might get stuck. 
                # Let's stick to simple moves: Add, Remove, Swap.
                
                # If we add it, it's an improvement in value.
                cur_selected[best_add_idx] = True
                cur_val += v[best_add_idx]
                remaining_cap -= w[best_add_idx]
                # Update index sets
                # Remove from un_sel, add to sel
                un_sel_idx = np.delete(un_sel_idx, np.where(un_sel_idx == best_add_idx)[0][0])
                sel_idx = np.append(sel_idx, best_add_idx)
                improved = True
                continue # Restart loop with new state

        # 2. Try SWAP: Replace one selected item with one unselected item
        # This is more complex. We look for a pair (i in sel, j in un_sel) such that:
        # w[j] <= w[i] + remaining_cap  (it fits after removing i)
        # v[j] > v[i]                   (it's better value)
        # Actually, condition is: cur_val - v[i] + v[j] > cur_val  => v[j] > v[i]
        # And weight condition: cur_weight - w[i] + w[j] <= cap_original => w[j] <= w[i] + remaining_cap
        
        if len(sel_idx) > 0 and len(un_sel_idx) > 0:
            # To keep it O(N^2) but small N (100 items), we can check pairs.
            # But 100*100 = 10k checks per iteration. 50 iterations = 500k checks. Fast enough in numpy?
            # Let's vectorize the swap search.
            
            # For each selected item i, what is the max weight of an unselected item j we can swap in?
            # w[j] <= w[i] + remaining_cap
            # And we want to maximize v[j] - v[i] (or just find any v[j] > v[i])
            
            best_gain = -np.inf
            best_i = -1
            best_j = -1
            
            # Iterate over selected items (usually small number)
            for i in sel_idx:
                wi = w[i]
                vi = v[i]
                # Max weight allowed for j
                max_wj = wi + remaining_cap
                
                # Find unselected items j where w[j] <= max_wj
                # And v[j] > vi (potential gain)
                # We want the one with highest v[j]
                
                mask_j = (w[un_sel_idx] <= max_wj)
                if not np.any(mask_j):
                    continue
                
                candidate_js = un_sel_idx[mask_j]
                if len(candidate_js) == 0:
                    continue
                
                # Among these, find the one with max value
                vals_j = v[candidate_js]
                best_j_local = candidate_js[np.argmax(vals_j)]
                best_val_j = v[best_j_local]
                
                gain = best_val_j - vi
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = best_j_local
            
            if best_i >= 0 and best_gain > 1e-9:
                # Perform swap
                cur_selected[best_i] = False
                cur_selected[best_j] = True
                cur_val += best_gain
                remaining_cap += w[best_i] - w[best_j]
                
                # Update index sets
                # Remove best_i from sel_idx
                sel_idx = np.delete(sel_idx, np.where(sel_idx == best_i)[0][0])
                # Remove best_j from un_sel_idx
                un_sel_idx = np.delete(un_sel_idx, np.where(un_sel_idx == best_j)[0][0])
                # Add best_j to sel_idx
                sel_idx = np.append(sel_idx, best_j)
                
                improved = True
                continue

        # 3. Try REMOVE: Remove a selected item if it allows adding a much better one?
        # This is effectively covered by Swap if we consider swapping with "nothing" (removing) 
        # but only if the freed space allows adding something else. 
        # Simple removal reduces value, so we only do it if it enables a subsequent add/swap that net positive.
        # The swap logic above handles replacing one with another. 
        # If removing i allows adding j AND k, that's a 2-move. 
        # For simplicity and speed, we stick to 1-move swaps and adds.
        
    return cur_selected, cur_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if remaining_capacity is None:
        return -1
    
    cap = float(remaining_capacity)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    
    if n == 0 or cap <= 0:
        return -1
        
    # Check if any item is feasible
    if not np.any(weights <= cap):
        return -1

    # 1. Generate initial solution using Density Greedy
    initial_sel, initial_val = _density_greedy_solution(cap, weights, values)
    
    # If no items selected, fallback to max density feasible item
    if not np.any(initial_sel):
        feasible = weights <= cap
        if not np.any(feasible):
            return -1
        # Pick max density among feasible
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(weights > 0, values / weights, 0.0)
        fd = np.where(feasible, density, -np.inf)
        return int(np.argmax(fd))

    # 2. Improve solution using Local Search (Hill Climbing)
    # We have plenty of time (120s limit, this is fast), so we can run a few iterations.
    # The local search modifies the mask. We just need the FIRST item picked by the final strategy.
    # BUT, the interface asks for "next item to add". 
    # The evaluator builds the solution iteratively.
    # If we return the first item of the *optimized* set, the evaluator will add it.
    # Then the state changes. 
    # Does the optimized set remain valid? No, because the set of available items changes.
    
    # CRITICAL INSIGHT:
    # The local search optimizes the *entire* remaining set.
    # If we pick the "best" first item from that optimized set, we are effectively committing to that path.
    # However, the optimized set might contain items that are not the "first" in a greedy sense.
    # The evaluator expects us to return an index.
    # If we return index `k` from the optimized set, the evaluator adds item `k`.
    # Then it calls us again with item `k` removed.
    
    # Strategy:
    # 1. Solve the full 0/1 knapsack approximately for the current state using Local Search starting from Density Greedy.
    # 2. From the resulting optimal(ish) set, pick the item that is "most critical" or simply the first one in some order?
    # Actually, any item in the optimal set is a valid "next" move if we plan to pick the rest later.
    # But if we pick item A, and the optimal set was {A, B, C}, and later we are asked to pick again,
    # we might pick B, then C.
    # The problem is: what if the optimal set is {A, B}, but after picking A, the optimal set for the remainder is {B, D}?
    # Then picking A was still good because B is still there.
    
    # However, a simpler and more robust heuristic for "select_next_item" when you have a full solver:
    # Pick the item from the optimized solution that has the highest "density" or "value"?
    # Or just pick the first one in the optimized solution?
    # Let's pick the item with the highest value among those in the optimized set.
    # Why? Because it's the most valuable component of the good solution.
    
    improved_sel, improved_val = _improve_solution(initial_sel, weights, values, cap)
    
    # If local search didn't improve, use initial
    if improved_val <= initial_val:
        final_sel = initial_sel
    else:
        final_sel = improved_sel
    
    # Get indices of selected items in the final solution
    selected_indices = np.where(final_sel)[0]
    
    if len(selected_indices) == 0:
        # Fallback: max density feasible
        feasible = weights <= cap
        if not np.any(feasible):
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(weights > 0, values / weights, 0.0)
        fd = np.where(feasible, density, -np.inf)
        return int(np.argmax(fd))
    
    # Choose which item to return.
    # Heuristic: Return the item with the highest value in the selected set.
    # Rationale: It's the "anchor" of the high-value solution.
    # Alternatively, return the one with highest density?
    # Let's try highest value.
    
    vals_in_sel = values[selected_indices]
    best_local_idx = np.argmax(vals_in_sel)
    return int(selected_indices[best_local_idx])

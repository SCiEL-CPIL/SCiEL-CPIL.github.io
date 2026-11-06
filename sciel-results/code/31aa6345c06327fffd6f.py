import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    cap = float(remaining_capacity)
    if cap <= 0:
        return -1
    
    w_orig = np.asarray(remaining_weights, dtype=np.float64)
    v_orig = np.asarray(remaining_values, dtype=np.float64)
    n_orig = len(w_orig)
    if n_orig == 0:
        return -1
    
    # Filter feasible
    mask = w_orig <= cap
    if not np.any(mask):
        return -1
    feasible_indices = np.where(mask)[0]
    w = w_orig[mask]
    v = v_orig[mask]
    n = len(w)
    
    # Sort by density descending
    densities = v / np.maximum(w, 1e-12)
    order = np.argsort(-densities)
    w = w[order]
    v = v[order]
    
    orig_ids = feasible_indices[order]
    
    # Time budget
    start_time = time.monotonic()
    deadline = start_time + 145.0  # 5s safety margin
    
    # --- Step 1: Greedy + local search to get a good incumbent ---
    best_val = 0.0
    best_sol = []  # list of indices in sorted (w, v) arrays
    
    # Greedy by density
    cur_cap = cap
    for i in range(n):
        if w[i] <= cur_cap:
            best_sol.append(i)
            best_val += v[i]
            cur_cap -= w[i]
    
    # 1-swap local refinement (a few passes)
    if best_sol:
        sol_set = set(best_sol)
        for _ in range(5):
            improved = False
            for remove_idx in best_sol[:]:
                new_cap = cap - sum(w[i] for i in best_sol if i != remove_idx)
                new_val = best_val - v[remove_idx]
                
                best_add = -1
                best_add_val = -1
                for add_idx in range(n):
                    if add_idx in sol_set or add_idx == remove_idx:
                        continue
                    if w[add_idx] <= new_cap:
                        if v[add_idx] > best_add_val:
                            best_add_val = v[add_idx]
                            best_add = add_idx
                
                if best_add != -1 and new_val + best_add_val > best_val + 1e-12:
                    best_sol.remove(remove_idx)
                    best_sol.append(best_add)
                    sol_set.discard(remove_idx)
                    sol_set.add(best_add)
                    best_val = new_val + best_add_val
                    improved = True
                    break
            if not improved:
                break
    
    # --- Step 2: B&B with iterative DFS ---
    # State: (idx, cap_cur, val_cur, path)
    # We use a stack to avoid recursion depth issues
    # Each stack entry: (idx, cap_cur, val_cur, path_tuple)
    # But tuples are expensive for 100 items. Let's use a different approach.
    # Actually, for n=100, recursion depth 100 is fine in Python (default limit ~1000).
    # The previous concern was unfounded. Let's use recursion but with tight time checks.
    
    nodes_visited = 0
    
    def bb(idx, cap_cur, val_cur, current_path):
        nonlocal best_val, best_sol, nodes_visited
        
        if time.monotonic() > deadline:
            return
        
        nodes_visited += 1
        
        if idx == n:
            if val_cur > best_val:
                best_val = val_cur
                best_sol = current_path[:]
            return
        
        # Bound: fractional knapsack from idx
        b_val = val_cur
        b_cap = cap_cur
        for j in range(idx, n):
            if w[j] <= b_cap:
                b_val += v[j]
                b_cap -= w[j]
            else:
                b_val += v[j] * (b_cap / w[j])
                break
        
        if b_val <= best_val + 1e-12:
            return
        
        # Branch: Take item idx (prefer taking to find good solutions early)
        if w[idx] <= cap_cur:
            current_path.append(idx)
            bb(idx + 1, cap_cur - w[idx], val_cur + v[idx], current_path)
            current_path.pop()
        
        if time.monotonic() > deadline:
            return
        
        # Branch: Skip item idx
        bb(idx + 1, cap_cur, val_cur, current_path)
    
    bb(0, cap, 0.0, [])
    
    # --- Step 3: Final local refinement on best solution ---
    if best_sol:
        sol_set = set(best_sol)
        for _ in range(3):
            improved = False
            for remove_idx in best_sol[:]:
                new_cap = cap - sum(w[i] for i in best_sol if i != remove_idx)
                new_val = best_val - v[remove_idx]
                
                best_add = -1
                best_add_val = -1
                for add_idx in range(n):
                    if add_idx in sol_set or add_idx == remove_idx:
                        continue
                    if w[add_idx] <= new_cap:
                        if v[add_idx] > best_add_val:
                            best_add_val = v[add_idx]
                            best_add = add_idx
                
                if best_add != -1 and new_val + best_add_val > best_val + 1e-12:
                    best_sol.remove(remove_idx)
                    best_sol.append(best_add)
                    sol_set.discard(remove_idx)
                    sol_set.add(best_add)
                    best_val = new_val + best_add_val
                    improved = True
                    break
            if not improved:
                break
    
    if not best_sol:
        # Fallback: pick highest value feasible item
        max_val_idx = int(np.argmax(v))
        return int(orig_ids[max_val_idx])
    
    # --- Step 4: Select the item to return ---
    # We need to pick one item from best_sol to "commit" to.
    # Strategy: pick the item with highest value (locks in value quickly).
    # Alternative: pick the item whose removal causes largest value drop.
    # Let's pick the highest value item.
    vals = np.array([v[i] for i in best_sol])
    max_pos = int(np.argmax(vals))
    chosen_sorted_idx = best_sol[max_pos]
    
    return int(orig_ids[chosen_sorted_idx])

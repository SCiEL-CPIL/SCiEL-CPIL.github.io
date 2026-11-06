import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    cap = float(remaining_capacity)
    if cap <= 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    n = len(w)
    if n == 0:
        return -1
    
    # Filter feasible
    mask = w <= cap
    if not np.any(mask):
        return -1
    w = w[mask]
    v = v[mask]
    n = len(w)
    
    # Sort by density descending
    densities = v / np.maximum(w, 1e-12)
    order = np.argsort(-densities)
    w = w[order]
    v = v[order]
    
    feasible_indices = np.where(mask)[0]
    orig_ids = feasible_indices[order]
    
    # Precompute suffix fractional knapsack bounds for O(1) bound lookup
    # suffix_bound[i] = max fractional value achievable from items i..n-1 given capacity
    # We compute this on the fly with memoization per capacity isn't feasible,
    # so we use a tighter bound: for each idx, the fractional bound assuming we
    # can take items from idx onward in density order (already sorted)
    
    start_time = time.monotonic()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    best_solution = []
    best_val = 0.0
    nodes_visited = 0
    timed_out = [False]
    
    def compute_fractional_bound(idx, cap_cur):
        """Compute fractional knapsack bound from item idx onward with capacity cap_cur."""
        b_val = 0.0
        b_cap = cap_cur
        for j in range(idx, n):
            if w[j] <= b_cap:
                b_val += v[j]
                b_cap -= w[j]
            else:
                b_val += v[j] * (b_cap / w[j])
                break
        return b_val
    
    def bb(idx, cap_cur, val_cur, current_path):
        nonlocal best_val, best_solution, nodes_visited, timed_out
        
        # Time check - hard deadline
        if time.monotonic() > deadline:
            timed_out[0] = True
            return
        
        nodes_visited += 1
        
        if idx == n:
            if val_cur > best_val:
                best_val = val_cur
                best_solution = current_path[:]
            return
        
        # Bound: fractional knapsack from idx
        b_val = val_cur + compute_fractional_bound(idx, cap_cur)
        
        if b_val <= best_val + 1e-12:
            return
        
        # Branch: Take item idx (prefer taking to find good solutions early)
        if w[idx] <= cap_cur:
            current_path.append(idx)
            bb(idx + 1, cap_cur - w[idx], val_cur + v[idx], current_path)
            current_path.pop()
        
        # Check time before skip branch
        if time.monotonic() > deadline:
            timed_out[0] = True
            return
        
        # Branch: Skip item idx
        bb(idx + 1, cap_cur, val_cur, current_path)
    
    # Run B&B
    bb(0, cap, 0.0, [])
    
    if not best_solution:
        # Fallback: greedy by density
        d = v / np.maximum(w, 1e-12)
        best_sorted_idx = int(np.argmax(d))
        return int(orig_ids[best_sorted_idx])
    
    # Local refinement: 1-swap and 2-swap local search
    sol_set = set(best_solution)
    
    def try_1_swap():
        """Try all 1-swap moves. Returns True if improvement found."""
        nonlocal best_solution, best_val, sol_set
        for remove_idx in list(best_solution):
            new_cap = cap - sum(w[i] for i in best_solution if i != remove_idx)
            new_val = best_val - v[remove_idx]
            
            best_add = -1
            best_add_val = -1
            for add_idx in range(n):
                if add_idx in sol_set:
                    continue
                if w[add_idx] <= new_cap:
                    if v[add_idx] > best_add_val:
                        best_add_val = v[add_idx]
                        best_add = add_idx
            
            if best_add != -1 and new_val + best_add_val > best_val + 1e-12:
                best_solution.remove(remove_idx)
                best_solution.append(best_add)
                sol_set.discard(remove_idx)
                sol_set.add(best_add)
                best_val = new_val + best_add_val
                return True
        return False
    
    def try_2_swap():
        """Try all 2-swap moves (remove 2, add 2). Returns True if improvement found."""
        nonlocal best_solution, best_val, sol_set
        sol_list = list(best_solution)
        m = len(sol_list)
        if m < 2:
            return False
        
        for i in range(m):
            for j in range(i + 1, m):
                rem_i = sol_list[i]
                rem_j = sol_list[j]
                new_cap = cap - sum(w[k] for k in best_solution if k not in (rem_i, rem_j))
                new_val = best_val - v[rem_i] - v[rem_j]
                
                # Find best pair to add
                best_pair = None
                best_pair_val = -1
                for a in range(n):
                    if a in sol_set:
                        continue
                    if w[a] > new_cap:
                        continue
                    rem_cap_a = new_cap - w[a]
                    for b in range(n):
                        if b in sol_set or b == a:
                            continue
                        if w[b] > rem_cap_a:
                            continue
                        pair_val = v[a] + v[b]
                        if pair_val > best_pair_val:
                            best_pair_val = pair_val
                            best_pair = (a, b)
                
                if best_pair is not None and new_val + best_pair_val > best_val + 1e-12:
                    a, b = best_pair
                    best_solution.remove(rem_i)
                    best_solution.remove(rem_j)
                    best_solution.append(a)
                    best_solution.append(b)
                    sol_set.discard(rem_i)
                    sol_set.discard(rem_j)
                    sol_set.add(a)
                    sol_set.add(b)
                    best_val = new_val + best_pair_val
                    return True
        return False
    
    # Multiple refinement passes
    for _ in range(5):
        improved = False
        if try_1_swap():
            improved = True
        if try_2_swap():
            improved = True
        if not improved:
            break
    
    # Pick the item with highest value from the solution
    sol_indices_in_sorted = best_solution
    vals = v[sol_indices_in_sorted]
    max_pos = int(np.argmax(vals))
    chosen_sorted_idx = sol_indices_in_sorted[max_pos]
    
    return int(orig_ids[chosen_sorted_idx])

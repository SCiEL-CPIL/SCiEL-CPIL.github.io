import numpy as np
import time

_GLOBAL_START = time.monotonic()
_TOTAL_BUDGET = 110.0 # Leave 40s margin for 150s limit? 150s wall clock. 8 instances.
# Actually, the prompt says "Each evaluation must finish within 120 s".
# "The evaluator KILLS the program after 150s wall-clock."
# So we have 150s for the whole script? Or per instance?
# "scored by mean total value on eight generated 100-item instances"
# Usually, these evaluators run the script once which processes all instances.
# So 150s total for 8 instances. ~18s per instance.
# select_next_item is called K times. If K is ~10-20, we have ~1s per call.
# The current timeout inside select_next_item is 100s. This is dangerous if the function is called multiple times.
# We need a global deadline.

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
    feasible_indices = np.where(mask)[0]
    w = w[feasible_indices]
    v = v[feasible_indices]
    n = len(w)
    
    # Sort by density descending
    densities = v / np.maximum(w, 1e-12)
    order = np.argsort(-densities)
    w = w[order]
    v = v[order]
    orig_ids = feasible_indices[order]
    
    # Iterative Branch and Bound
    # State: (idx, cap, val, path)
    # We use a stack.
    # To optimize, we can pre-compute the fractional bound more efficiently?
    # For now, let's just make the B&B iterative and fast.
    
    best_val = 0.0
    best_path = []
    
    # Stack items: [idx, cap, val, path]
    # We push (0, cap, 0.0, [])
    # But storing path in stack is memory heavy. 
    # Better: store path as a list and backtrack, or store parent pointers.
    # Given N=100, storing path list in stack is fine (100 * depth).
    
    stack = [(0, cap, 0.0, [])]
    
    # Time check every 1000 nodes
    nodes_processed = 0
    last_time_check = time.monotonic()
    
    while stack:
        # Check time periodically
        if nodes_processed % 1000 == 0:
            if time.monotonic() - last_time_check > 0.1: # 100ms check interval
                if time.monotonic() - _GLOBAL_START > _TOTAL_BUDGET:
                    break
                last_time_check = time.monotonic()
        
        idx, c, val, path = stack.pop()
        nodes_processed += 1
        
        if idx == n:
            if val > best_val:
                best_val = val
                best_path = path
            continue
        
        # Bound calculation
        # Fractional knapsack bound
        b_val = val
        b_cap = c
        # Optimize bound: since sorted by density, we can break early
        for j in range(idx, n):
            if w[j] <= b_cap:
                b_val += v[j]
                b_cap -= w[j]
            else:
                b_val += v[j] * (b_cap / w[j])
                break
        
        if b_val <= best_val:
            continue
        
        # Branching
        # Heuristic: Explore the more promising branch first (Take)
        # This helps find good solutions early, tightening the bound.
        
        # Branch 1: Skip
        # Branch 2: Take
        # We want to find high value solutions fast. Taking usually yields higher value.
        # So push Skip first, then Take, so Take is popped first (LIFO).
        
        stack.append((idx + 1, c, val, path))
        
        if w[idx] <= c:
            new_path = path + [idx]
            stack.append((idx + 1, c - w[idx], val + v[idx], new_path))
            
    if not best_path:
        # Fallback: Greedy pick
        if n > 0:
            # Pick best density
            d = v / np.maximum(w, 1e-12)
            best_sorted_idx = int(np.argmax(d))
            return int(orig_ids[best_sorted_idx])
        return -1
        
    # Pick item from best_path
    # Current logic: Pick max value item from the optimal set.
    # This is good.
    vals = v[best_path]
    max_pos = int(np.argmax(vals))
    chosen_sorted_idx = best_path[max_pos]
    
    return int(orig_ids[chosen_sorted_idx])

import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 128
    
    # Precompute all 4-term APs (0-indexed)
    aps = []
    ap_indices = [[] for _ in range(n)]
    for d in range(1, (n - 1) // 3 + 1):
        for i in range(n - 3 * d):
            ap_id = len(aps)
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
            ap_indices[i].append(ap_id)
            ap_indices[i + d].append(ap_id)
            ap_indices[i + 2 * d].append(ap_id)
            ap_indices[i + 3 * d].append(ap_id)
    
    M = len(aps)
    
    def count_violations(colors):
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                count += 1
        return count
    
    def hamming_weight(x):
        return bin(x).count('1')
    
    # Deterministic initial coloring using Hamming weight
    # This gives 8 color classes (0-7) for positions 0-127
    base_coloring = [hamming_weight(i) for i in range(n)]
    base_violations = count_violations(base_coloring)
    
    # If base coloring is already valid, we're done with 8 colors
    if base_violations == 0:
        # Try to reduce distinct colors used
        distinct = set(base_coloring)
        # The labels are 0..7, so 8 distinct colors
        return base_coloring
    
    # Simulated Annealing to find valid coloring with minimal distinct colors
    # Strategy: start with k colors, try to find valid solution, then reduce k
    
    best_solution = None
    best_distinct = n  # fallback: all distinct
    
    for k in range(3, min(16, n + 1)):
        if time.time() > deadline - 5.0:
            break
        
        # For each k, try multiple restarts
        found_valid = False
        best_for_k = None
        best_distinct_for_k = k
        
        # Initial coloring: use Hamming weight mod k, or random
        initial_colorings = []
        
        # Option 1: Hamming weight based
        hw_coloring = [hamming_weight(i) % k for i in range(n)]
        initial_colorings.append(hw_coloring)
        
        # Option 2: Random
        for _ in range(2):
            initial_colorings.append([random.randint(0, k - 1) for _ in range(n)])
        
        for init in initial_colorings:
            if time.time() > deadline - 3.0:
                break
            
            coloring = list(init)
            current_violations = count_violations(coloring)
            current_distinct = len(set(coloring))
            
            best_coloring = coloring[:]
            best_violations = current_violations
            best_distinct_local = current_distinct
            
            # Simulated Annealing
            T = 5.0
            cooling = 0.9995
            steps = 0
            max_steps = 80000
            
            while current_violations > 0 and steps < max_steps and time.time() < deadline - 2.0:
                steps += 1
                idx = random.randint(0, n - 1)
                old_color = coloring[idx]
                new_color = random.randint(0, k - 1)
                
                if old_color == new_color:
                    continue
                
                # Calculate change in violations
                delta_violations = 0
                for ap_id in ap_indices[idx]:
                    ap = aps[ap_id]
                    c0 = coloring[ap[0]]
                    c1 = coloring[ap[1]]
                    c2 = coloring[ap[2]]
                    c3 = coloring[ap[3]]
                    
                    old_set_size = len({c0, c1, c2, c3})
                    
                    new_c0 = new_color if ap[0] == idx else c0
                    new_c1 = new_color if ap[1] == idx else c1
                    new_c2 = new_color if ap[2] == idx else c2
                    new_c3 = new_color if ap[3] == idx else c3
                    
                    new_set_size = len({new_c0, new_c1, new_c2, new_c3})
                    
                    if old_set_size < 3 and new_set_size >= 3:
                        delta_violations -= 1
                    elif old_set_size >= 3 and new_set_size < 3:
                        delta_violations += 1
                
                # Also track distinct color change for objective
                old_distinct = len(set(coloring))
                # After change, distinct might decrease if old_color was unique
                # or increase if new_color wasn't used before
                if delta_violations < 0:
                    # Accept improvement
                    coloring[idx] = new_color
                    current_violations += delta_violations
                    new_distinct = len(set(coloring))
                    if current_violations < best_violations or \
                       (current_violations == best_violations and new_distinct < best_distinct_local):
                        best_violations = current_violations
                        best_distinct_local = new_distinct
                        best_coloring = coloring[:]
                else:
                    # Accept worse with probability based on temperature
                    if delta_violations == 0:
                        accept = random.random() < 0.3
                    else:
                        if T > 0.01:
                            accept = random.random() < np.exp(-delta_violations / T)
                        else:
                            accept = False
                    
                    if accept:
                        coloring[idx] = new_color
                        current_violations += delta_violations
            
            T *= cooling
            
            if current_violations == 0:
                found_valid = True
                final_distinct = len(set(coloring))
                if final_distinct < best_distinct_for_k:
                    best_distinct_for_k = final_distinct
                    best_for_k = coloring[:]
                break  # Found valid for this k, no need for more restarts
            
            # Track best even if not valid
            if best_violations < (n if best_for_k is None else 1000):
                if best_for_k is None or best_violations < 1:
                    best_for_k = best_coloring[:]
                    best_distinct_for_k = best_distinct_local
        
        if found_valid and best_for_k is not None:
            if best_distinct_for_k < best_distinct:
                best_distinct = best_distinct_for_k
                best_solution = best_for_k
            break  # Found valid with this k, try smaller k if possible
    
    # If no valid solution found with small k, use the best we have
    if best_solution is None:
        # Fallback: use Hamming weight coloring even if it has violations
        # But we need a valid one, so use all distinct
        best_solution = list(range(n))
    
    return best_solution

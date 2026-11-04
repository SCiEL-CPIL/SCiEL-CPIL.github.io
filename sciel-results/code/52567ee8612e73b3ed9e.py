import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initial Solution: Rank Matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    time_limit = 115.0
    
    # Tabu Search Parameters
    tabu_tenure = 20
    tabu_list = {} # move (i,j) -> iteration when it becomes non-tabu
    current_iter = 0
    
    # Precompute row sums of a? Not needed for delta.
    
    def get_delta(i, j, p):
        li = p[i]
        lj = p[j]
        
        # Vectorized delta components
        # D1: a[i, :] * (b[lj, p] - b[li, p])
        d1 = a[i, :] * (b[lj, p] - b[li, p])
        d1[i] = 0
        d1[j] = 0
        
        # D2: a[j, :] * (b[li, p] - b[lj, p])
        d2 = a[j, :] * (b[li, p] - b[lj, p])
        d2[i] = 0
        d2[j] = 0
        
        # D3: a[:, i] * (b[p, lj] - b[p, li])
        d3 = a[:, i] * (b[p, lj] - b[p, li])
        d3[i] = 0
        d3[j] = 0
        
        # D4: a[:, j] * (b[p, li] - b[p, lj])
        d4 = a[:, j] * (b[p, li] - b[p, lj])
        d4[i] = 0
        d4[j] = 0
        
        sum_delta = np.sum(d1) + np.sum(d2) + np.sum(d3) + np.sum(d4)
        
        # Interaction term
        delta_ij = a[i, j] * (b[lj, li] - b[li, lj]) + a[j, i] * (b[li, lj] - b[lj, li])
        
        return sum_delta + delta_ij

    while time.time() - start_time < time_limit:
        current_iter += 1
        
        best_delta = -np.inf
        best_i = -1
        best_j = -1
        best_new_cost = current_cost
        
        # Evaluate all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dc = get_delta(i, j, p)
                new_cost = current_cost + dc
                
                # Check tabu status
                key = (i, j)
                is_tabu = key in tabu_list and tabu_list[key] > current_iter
                is_aspiration = new_cost < best_cost
                
                if is_tabu and not is_aspiration:
                    continue
                    
                # We want the best move (lowest cost)
                # If all moves are tabu and not aspiration, we might need to relax
                # But for now, if no valid move found, we might pick the best tabu one?
                # Standard TS: pick best non-tabu. If none, pick best tabu (or restart).
                
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_delta = dc
                    best_i = i
                    best_j = j
                    
        if best_i == -1:
            # No valid move found (all tabu and not aspiration)
            # Clear tabu list and pick best move regardless
            tabu_list.clear()
            for i in range(n):
                for j in range(i + 1, n):
                    dc = get_delta(i, j, p)
                    new_cost = current_cost + dc
                    if new_cost < best_new_cost:
                        best_new_cost = new_cost
                        best_delta = dc
                        best_i = i
                        best_j = j
                        is_aspiration = new_cost < best_cost # Update aspiration check for this specific move if we restart? 
                        # Actually, if we cleared tabu, it's not tabu anymore.
            
            if best_i == -1:
                break # Should not happen
        
        # Apply move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost = best_new_cost
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Update tabu list
        key = (min(best_i, best_j), max(best_i, best_j))
        tabu_list[key] = current_iter + tabu_tenure
        
        # Cleanup old tabu entries to save memory (optional, dict is small)
        if len(tabu_list) > 1000:
            # Remove oldest
            # This is inefficient, but list size is bounded by tenure * 2 * n? No, unique pairs.
            # Just let it grow or clear periodically.
            pass

    return best_p.tolist()

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Phase 1: Greedy initialization
    # Sort customers by demand descending to place large ones first
    order = np.argsort(-demands)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    count = np.zeros(n_facilities, dtype=int)
    
    for i in order:
        best_f = -1
        best_score = np.inf
        for f in range(n_facilities):
            if load[f] + demands[i] > capacities[f]:
                continue
            # Simple greedy: minimize assignment cost + marginal open cost
            if count[f] == 0:
                score = assign_cost[i, f] + open_cost[f]
            else:
                score = assign_cost[i, f]
            if score < best_score:
                best_score = score
                best_f = f
        if best_f == -1:
            # Fallback: find feasible
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    best_f = f
                    break
            if best_f == -1:
                # Should not happen if problem is feasible
                best_f = np.argmax(capacities - load)
        result[i] = best_f
        load[best_f] += demands[i]
        count[best_f] += 1
    
    # Calculate initial cost
    open_mask = (count > 0)
    total_cost = np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(n_customers), result])
    
    # Phase 2: Local Search with 1-Opt and 2-Opt
    # We iterate until no improvement is found in a full pass
    # To handle the open cost changes in 1-Opt, we need to be careful.
    # 2-Opt swaps between open facilities do not change open costs.
    
    max_passes = 100
    for _ in range(max_passes):
        improved = False
        
        # --- 1-Opt Moves ---
        # Try moving each customer i to facility f != result[i]
        for i in range(n_customers):
            old_f = result[i]
            best_new_f = -1
            best_delta = 0.0
            current_assign_cost_i = assign_cost[i, old_f]
            
            for f in range(n_facilities):
                if f == old_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                # Calculate delta
                # Assignment cost change
                delta_assign = assign_cost[i, f] - current_assign_cost_i
                
                # Open cost change
                delta_open = 0.0
                if count[old_f] == 1:
                    delta_open -= open_cost[old_f] # Close old
                if count[f] == 0:
                    delta_open += open_cost[f]    # Open new
                
                delta = delta_assign + delta_open
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_new_f = f
            
            if best_new_f != -1:
                # Apply move
                load[old_f] -= demands[i]
                load[best_new_f] += demands[i]
                count[old_f] -= 1
                count[best_new_f] += 1
                result[i] = best_new_f
                total_cost += best_delta
                improved = True
        
        # --- 2-Opt Swaps ---
        # Try swapping customer i at f_i with customer j at f_j (f_i != f_j)
        # We can iterate i < j to avoid duplicates
        for i in range(n_customers):
            f_i = result[i]
            if count[f_i] == 0: continue # Should not happen
            
            for j in range(i + 1, n_customers):
                f_j = result[j]
                if f_i == f_j:
                    continue
                
                # Check capacity after swap
                # f_i loses i, gains j
                if load[f_i] - demands[i] + demands[j] > capacities[f_i]:
                    continue
                # f_j loses j, gains i
                if load[f_j] - demands[j] + demands[i] > capacities[f_j]:
                    continue
                
                # Calculate delta
                # Assignment cost change
                delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                
                # Open cost change: 0 (both facilities remain non-empty if they were non-empty)
                # Wait, if count[f_i] == 1, it becomes 1 (swapped). So open cost unchanged.
                delta_open = 0.0
                
                delta = delta_assign + delta_open
                if delta < -1e-9:
                    # Apply swap
                    load[f_i] -= demands[i]
                    load[f_i] += demands[j]
                    load[f_j] -= demands[j]
                    load[f_j] += demands[i]
                    # Counts remain same
                    result[i] = f_j
                    result[j] = f_i
                    total_cost += delta
                    improved = True
                    # Note: We don't break out of inner loop immediately because
                    # another swap might be better, but standard steepest descent
                    # would pick the best. For speed, we take the first improving swap
                    # or we can track best. Given N=100, N^2/2 = 5000 checks.
                    # Taking first improvement is faster per iteration.
        
        if not improved:
            break
            
    return result
# EVOLVE-BLOCK-END

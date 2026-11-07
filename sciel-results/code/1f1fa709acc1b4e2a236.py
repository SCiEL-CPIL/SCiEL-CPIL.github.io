# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_customers = len(demands)
    n_facilities = len(capacities)

    # 1. Initial Solution: Greedy
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
            if count[f] == 0:
                score = assign_cost[i, f] + open_cost[f]
            else:
                score = assign_cost[i, f] + open_cost[f] / (count[f] + 1)
            if score < best_score:
                best_score = score
                best_f = f
        if best_f == -1:
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    best_f = f
                    break
            if best_f == -1:
                best_f = int(np.argmax(capacities - load))
        result[i] = best_f
        load[best_f] += demands[i]
        count[best_f] += 1

    open_mask = (count > 0)
    current_cost = float(np.sum(open_cost[open_mask])) + float(np.sum(assign_cost[np.arange(n_customers), result]))
    best_cost = current_cost
    best_result = result.copy()

    # 2. Tabu Search (Vectorized)
    max_iters = 20000
    tabu_tenure = 15
    
    # tabu_until[i, f] = iteration when tabu expires for moving i to f
    # We tabu the SOURCE facility to prevent reversal.
    # So if we move i from f_old to f_new, we set tabu_until[i, f_old] = it + tenure.
    tabu_until = np.full((n_customers, n_facilities), -1, dtype=int)
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > 110:
            break
            
        best_move_i = -1
        best_move_f = -1
        best_delta = 0.0
        best_is_aspiration = False

        for i in range(n_customers):
            f_old = result[i]
            d_i = demands[i]
            
            # Capacity Mask
            cap_ok = (load + d_i <= capacities)
            cap_ok[f_old] = False
            
            # Tabu Mask
            tabu_ok = tabu_until[i, :] <= it
            
            valid_mask = cap_ok & tabu_ok
            
            if not np.any(valid_mask):
                continue
                
            # Cost Deltas
            # delta_assign = assign_cost[i, :] - assign_cost[i, f_old]
            # We only care about valid indices, but computing for all is fast.
            delta_assign = assign_cost[i, :] - assign_cost[i, f_old]
            
            # Open Cost Delta
            # Base is 0.
            # If count[f_old] == 1, we save open_cost[f_old] for any move away.
            # If count[f_new] == 0, we pay open_cost[f_new].
            
            delta_open = np.zeros(n_facilities)
            
            if count[f_old] == 1:
                # Subtract open_cost[f_old] from all valid destinations
                delta_open[valid_mask] -= open_cost[f_old]
                
            # Add open_cost[f] for f where count[f] == 0 and valid
            zero_count_mask = (count == 0)
            add_open_mask = valid_mask & zero_count_mask
            if np.any(add_open_mask):
                delta_open[add_open_mask] += open_cost[add_open_mask]
                
            total_delta = delta_assign + delta_open
            total_delta[~valid_mask] = np.inf
            
            min_idx = np.argmin(total_delta)
            min_val = total_delta[min_idx]
            
            if min_val < np.inf:
                new_cost = current_cost + min_val
                if new_cost < best_cost:
                    best_move_i = i
                    best_move_f = min_idx
                    best_delta = min_val
                    best_is_aspiration = True
                    break
                elif min_val < best_delta:
                    best_move_i = i
                    best_move_f = min_idx
                    best_delta = min_val
                    best_is_aspiration = False

        if best_move_i == -1:
            # No valid move found (all tabu or infeasible).
            # Fallback: Find best feasible move ignoring tabu.
            best_delta = np.inf
            for i in range(n_customers):
                f_old = result[i]
                d_i = demands[i]
                cap_ok = (load + d_i <= capacities)
                cap_ok[f_old] = False
                if not np.any(cap_ok):
                    continue
                delta_assign = assign_cost[i, :] - assign_cost[i, f_old]
                delta_open = np.zeros(n_facilities)
                if count[f_old] == 1:
                    delta_open[cap_ok] -= open_cost[f_old]
                zero_count_mask = (count == 0)
                add_open_mask = cap_ok & zero_count_mask
                if np.any(add_open_mask):
                    delta_open[add_open_mask] += open_cost[add_open_mask]
                total_delta = delta_assign + delta_open
                total_delta[~cap_ok] = np.inf
                min_idx = np.argmin(total_delta)
                min_val = total_delta[min_idx]
                if min_val < best_delta:
                    best_delta = min_val
                    best_move_i = i
                    best_move_f = min_idx
            
            if best_move_i == -1:
                break

        # Apply Move
        i = best_move_i
        f_new = best_move_f
        f_old = result[i]
        
        # Update Tabu: Tabu the source facility
        tabu_until[i, f_old] = it + tabu_tenure
        
        # Update State
        result[i] = f_new
        load[f_old] -= demands[i]
        load[f_new] += demands[i]
        count[f_old] -= 1
        count[f_new] += 1
        
        # Update Cost
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_result = result.copy()

    return best_result
# EVOLVE-BLOCK-END

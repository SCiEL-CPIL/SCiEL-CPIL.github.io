"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    timeout = 115.0  # Safety margin below 120s
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays for ease of use
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # 1. Greedy Initialization: Largest Demand First
    # Sort indices by demand descending
    order = np.argsort(-demands)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    
    for i in order:
        d = demands[i]
        remaining_cap = capacities - load
        feasible_mask = remaining_cap >= d
        if not np.any(feasible_mask):
            # Fallback: pick the one with most remaining capacity
            f = np.argmax(remaining_cap)
        else:
            # Among feasible, pick the one with min assignment cost
            costs = np.where(feasible_mask, assign_cost[i], np.inf)
            f = np.argmin(costs)
        
        result[i] = f
        load[f] += d

    # Calculate initial cost
    def calc_total_cost(result, load):
        open_cost_sum = np.sum(open_cost[load > 0])
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_cost_sum + assign_cost_sum

    current_cost = calc_total_cost(result, load)
    best_cost = current_cost
    best_result = result.copy()
    best_load = load.copy()
    
    # 2. Local Search: Best Improvement with 1-move and 2-move (swap)
    # We will iterate until no improvement or time limit.
    
    # Helper to check time
    def time_exceeded():
        return time.time() - start_time > timeout

    improved = True
    while improved:
        if time_exceeded():
            break
        
        improved = False
        best_move_type = None
        best_move_data = None
        best_delta = 0.0
        
        # --- Phase A: Check 1-moves (Relocation) ---
        # For each customer i, try moving to any other facility f_new
        for i in range(n_customers):
            if time_exceeded():
                break
            
            f_old = result[i]
            d = demands[i]
            
            # Check all other facilities
            for f_new in range(n_facilities):
                if f_new == f_old:
                    continue
                
                # Feasibility
                if load[f_new] + d > capacities[f_new]:
                    continue
                
                # Cost Delta
                delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                delta_open = 0.0
                
                # Check if f_old becomes empty
                if load[f_old] == d:
                    delta_open -= open_cost[f_old]
                
                # Check if f_new was empty
                if load[f_new] == 0:
                    delta_open += open_cost[f_new]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_move_type = 1
                    best_move_data = (i, f_old, f_new)
        
        # --- Phase B: Check 2-moves (Swap) ---
        # For each pair of customers (i, j) at different facilities, try swapping them
        # To optimize, we can iterate i, then j > i, and only consider if result[i] != result[j]
        
        # Optimization: Group customers by facility to reduce checks? 
        # N=100 is small enough that O(N^2) is fine (10k checks).
        
        for i in range(n_customers):
            if time_exceeded():
                break
            
            f_i = result[i]
            d_i = demands[i]
            
            for j in range(i + 1, n_customers):
                if time_exceeded():
                    break
                
                f_j = result[j]
                d_j = demands[j]
                
                if f_i == f_j:
                    continue
                
                # Feasibility of swap
                # New load for f_i: load[f_i] - d_i + d_j
                # New load for f_j: load[f_j] - d_j + d_i
                new_load_i = load[f_i] - d_i + d_j
                new_load_j = load[f_j] - d_j + d_i
                
                if new_load_i > capacities[f_i] or new_load_j > capacities[f_j]:
                    continue
                
                # Cost Delta
                delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                               (assign_cost[i, f_i] + assign_cost[j, f_j])
                
                # Open cost delta
                # Old open cost contribution from f_i and f_j
                old_open = (open_cost[f_i] if load[f_i] > 0 else 0.0) + \
                           (open_cost[f_j] if load[f_j] > 0 else 0.0)
                
                # New open cost contribution from f_i and f_j
                new_open = (open_cost[f_i] if new_load_i > 0 else 0.0) + \
                           (open_cost[f_j] if new_load_j > 0 else 0.0)
                
                delta_open = new_open - old_open
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_move_type = 2
                    best_move_data = (i, j, f_i, f_j)
        
        # Apply best move if found
        if best_move_type is not None:
            if best_move_type == 1:
                i, f_old, f_new = best_move_data
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                result[i] = f_new
            else: # best_move_type == 2
                i, j, f_i, f_j = best_move_data
                # Swap i and j
                d_i = demands[i]
                d_j = demands[j]
                load[f_i] -= d_i
                load[f_j] -= d_j
                load[f_i] += d_j
                load[f_j] += d_i
                result[i] = f_j
                result[j] = f_i
            
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = result.copy()
                best_load = load.copy()
            improved = True
            
            if time_exceeded():
                break
        else:
            break

    # 3. If time permits, try a few random restarts or perturbations?
    # Given the strong 2-swap performance, sticking to the best local optimum found
    # is usually safer than risking a worse solution from a perturbation that doesn't recover.
    # However, if we have lots of time left, we could try diversification.
    # For now, return the best result from the 1+2 neighborhood local search.
    
    # Ensure valid return type
    return best_result.astype(int)
# EVOLVE-BLOCK-END

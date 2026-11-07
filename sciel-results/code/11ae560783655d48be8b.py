"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    timeout = 145.0  # Safety margin below 150s limit
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays for ease of use
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # Helper to calculate total cost
    def calc_total_cost(result, load):
        open_cost_sum = np.sum(open_cost[load > 0])
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_cost_sum + assign_cost_sum

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

    current_cost = calc_total_cost(result, load)
    best_cost = current_cost
    best_result = result.copy()
    best_load = load.copy()
    
    # 2. Local Search: Best Improvement
    # We will repeat best-improvement moves until no improvement or time limit.
    improved = True
    while improved:
        if time.time() - start_time > timeout - 5:
            break
        improved = False
        best_move = None
        best_delta = 0.0
        
        for i in range(n_customers):
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
                    best_move = (i, f_old, f_new)
        
        if best_move is not None:
            i, f_old, f_new = best_move
            # Apply move
            load[f_old] -= demands[i]
            load[f_new] += demands[i]
            result[i] = f_new
            
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = result.copy()
                best_load = load.copy()
            improved = True
        else:
            break

    # 3. Second Phase: First Improvement with Randomization (if time permits)
    # This attempts to escape local minima found by the best-improvement phase.
    if time.time() - start_time < timeout - 10:
        # Reset to best result found
        result = best_result.copy()
        load = best_load.copy()
        current_cost = best_cost
        
        # Use first-improvement to escape local minima
        np.random.seed(42)
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() - start_time > timeout - 5:
                break
            
            # Shuffle order for diversification
            perm = np.random.permutation(n_customers)
            found_improvement = False
            
            for i in perm:
                if time.time() - start_time > timeout - 5:
                    break
                
                f_old = result[i]
                d = demands[i]
                
                # Find all feasible new facilities
                feasible = []
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    if load[f_new] + d <= capacities[f_new]:
                        feasible.append(f_new)
                
                if not feasible:
                    continue
                
                # Calculate deltas for all feasible
                best_delta_local = 0.0
                best_f_local = f_old
                
                for f_new in feasible:
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    delta_open = 0.0
                    
                    if load[f_old] == d:
                        delta_open -= open_cost[f_old]
                    
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta_local:
                        best_delta_local = total_delta
                        best_f_local = f_new
                
                if best_f_local != f_old:
                    # Apply the best improvement found for this customer in this pass
                    load[f_old] -= demands[i]
                    load[best_f_local] += demands[i]
                    result[i] = best_f_local
                    current_cost += best_delta_local
                    found_improvement = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = result.copy()
                        best_load = load.copy()
            
            if not found_improvement:
                break

    # Return the best result found
    # Ensure integers
    return best_result.astype(int)
# EVOLVE-BLOCK-END

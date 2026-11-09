import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Precompute sorted facilities for each customer
    # sorted_facilities[i] contains facility indices sorted by assign_cost[i][f]
    sorted_facilities = np.argsort(assign_cost, axis=1)
    
    best_assignment = None
    best_cost = np.inf
    
    # Total time limit
    total_time_limit = 110.0
    start_global = time.time()
    
    # Number of restarts
    # We can adjust this based on time, but 10 is a good start.
    # If each restart takes 5s, 10 * 5 = 50s.
    # If each takes 1s, we can do more.
    # Let's do up to 20 restarts or until time runs out.
    
    for restart in range(20):
        if time.time() - start_global > total_time_limit:
            break
            
        # 1. Initialize
        assignment = np.zeros(n_customers, dtype=int)
        
        # Strategy: Random tie-breaking in nearest assignment
        # Add small random noise to costs for initialization only?
        # Or just random assignment and fix?
        # Random assignment is bad.
        # Let's use nearest with random tie-break.
        
        # To make it faster, we can just use the nearest facility, 
        # and if there's a tie, pick randomly.
        # But np.argmin doesn't do random tie-break easily.
        # We can add small random noise to assign_cost for init.
        
        noisy_costs = assign_cost + np.random.uniform(0, 1, assign_cost.shape)
        assignment = np.argmin(noisy_costs, axis=1)
        
        # 2. Fix Capacity Violations
        # This part is crucial.
        # We need to ensure feasibility.
        
        # Helper to fix
        def fix_capacities(assignment, load):
            # load is passed in to avoid recalculation
            for _ in range(n_customers * 2): # Max moves
                over_cap = load > capacities
                if not np.any(over_cap):
                    return True
                
                # Pick the facility with highest excess
                excess = np.maximum(0, load - capacities)
                f_over = np.argmax(excess)
                
                if excess[f_over] == 0:
                    break # Should not happen if over_cap is true
                
                # Find customers in f_over
                customers_in_f = np.where(assignment == f_over)[0]
                if len(customers_in_f) == 0:
                    break
                    
                # Sort by demand ascending (move smallest first)
                # This helps fit them into remaining spaces
                customers_in_f = customers_in_f[np.argsort(demands[customers_in_f])]
                
                moved = False
                for i in customers_in_f:
                    # Find best f' for i
                    # f' != f_over
                    # load[f'] + demands[i] <= capacities[f']
                    
                    # We can use sorted_facilities[i]
                    for f_new in sorted_facilities[i]:
                        if f_new == f_over:
                            continue
                        if load[f_new] + demands[i] <= capacities[f_new]:
                            # Move
                            assignment[i] = f_new
                            load[f_over] -= demands[i]
                            load[f_new] += demands[i]
                            moved = True
                            break
                    if moved:
                        break
                
                if not moved:
                    return False # Failed to fix
            return True

        load = np.bincount(assignment, minlength=n_facilities)
        if not fix_capacities(assignment, load):
            # If fix fails, try a different init or skip
            # This shouldn't happen if total capacity is sufficient
            continue
            
        # 3. Local Search
        # Move and Swap
        # Time limit per restart: 5 seconds
        restart_start = time.time()
        restart_limit = 5.0
        
        current_cost = np.sum(assign_cost[np.arange(n_customers), assignment]) + \
                       np.sum(open_cost[np.unique(assignment)])
        
        improved = True
        while improved:
            improved = False
            if time.time() - restart_start > restart_limit:
                break
                
            # Try Moves
            for i in range(n_customers):
                if time.time() - restart_start > restart_limit:
                    break
                f_old = assignment[i]
                for f_new in sorted_facilities[i]:
                    if f_new == f_old:
                        continue
                    if load[f_new] + demands[i] <= capacities[f_new]:
                        delta = assign_cost[i, f_new] - assign_cost[i, f_old]
                        
                        # Open cost changes
                        if load[f_old] == demands[i]:
                            delta -= open_cost[f_old]
                        if load[f_new] == 0:
                            delta += open_cost[f_new]
                            
                        if delta < 0:
                            assignment[i] = f_new
                            load[f_old] -= demands[i]
                            load[f_new] += demands[i]
                            current_cost += delta
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue
                
            # Try Swaps
            for i in range(n_customers):
                if time.time() - restart_start > restart_limit:
                    break
                f_i = assignment[i]
                for j in range(i + 1, n_customers):
                    if time.time() - restart_start > restart_limit:
                        break
                    f_j = assignment[j]
                    if f_i == f_j:
                        continue
                    
                    new_load_i = load[f_i] - demands[i] + demands[j]
                    new_load_j = load[f_j] - demands[j] + demands[i]
                    
                    if new_load_i <= capacities[f_i] and new_load_j <= capacities[f_j]:
                        delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                                       (assign_cost[i, f_i] + assign_cost[j, f_j])
                        
                        old_open_i = 1 if load[f_i] > 0 else 0
                        old_open_j = 1 if load[f_j] > 0 else 0
                        new_open_i = 1 if new_load_i > 0 else 0
                        new_open_j = 1 if new_load_j > 0 else 0
                        
                        delta_open = (new_open_i - old_open_i) * open_cost[f_i] + \
                                     (new_open_j - old_open_j) * open_cost[f_j]
                        
                        total_delta = delta_assign + delta_open
                        
                        if total_delta < 0:
                            assignment[i] = f_j
                            assignment[j] = f_i
                            load[f_i] = new_load_i
                            load[f_j] = new_load_j
                            current_cost += total_delta
                            improved = True
                            break
                if improved:
                    break
            
        # End Local Search
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_assignment = assignment.copy()
            
    # If no valid solution found (should not happen), return a fallback
    if best_assignment is None:
        # Fallback to greedy
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        return result

    return best_assignment

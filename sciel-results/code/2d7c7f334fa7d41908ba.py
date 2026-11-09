"""Combined program for Capacitated Facility Location."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    def compute_total_cost(assignment, loads):
        total = 0.0
        used = set(assignment.tolist())
        for f in used:
            total += open_cost[f]
        total += np.sum(assign_cost[np.arange(n_customers), assignment])
        return total
    
    def repair(assignment, loads):
        """Fix capacity violations by moving customers to feasible facilities."""
        for _ in range(100):
            overloaded = np.where(loads > capacities)[0]
            if len(overloaded) == 0:
                break
            
            f_over = overloaded[np.argmax(loads[overloaded] - capacities[overloaded])]
            customers_in_f = [i for i in range(n_customers) if assignment[i] == f_over]
            if not customers_in_f:
                continue
                
            # Sort by demand ascending (easiest to move)
            customers_in_f.sort(key=lambda x: demands[x])
            
            moved = False
            for i in customers_in_f:
                # Find feasible facility with min cost
                feasible = []
                for f in range(n_facilities):
                    if f == f_over:
                        continue
                    if loads[f] + demands[i] <= capacities[f]:
                        feasible.append(f)
                
                if feasible:
                    best_f = min(feasible, key=lambda f: assign_cost[i, f])
                    loads[f_over] -= demands[i]
                    loads[best_f] += demands[i]
                    assignment[i] = best_f
                    moved = True
                    break
            
            if not moved:
                break
        return assignment, loads

    # 1. Multiple Randomized Greedy Initializations
    best_init_cost = float('inf')
    best_init_assignment = None
    best_init_loads = None
    
    # Sort customers by demand descending for greedy
    order = np.argsort(-demands)
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        result = np.full(n_customers, -1, dtype=np.int64)
        load = np.zeros(n_facilities, dtype=np.int64)
        
        feasible_flag = True
        for i in order:
            # Find feasible facilities
            feasible = []
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    feasible.append(f)
            
            if not feasible:
                feasible_flag = False
                break
            
            if len(feasible) == 1:
                best_f = feasible[0]
            else:
                # Add noise to costs to explore different solutions
                # Noise scale depends on cost magnitude
                costs = assign_cost[i, feasible]
                noise = np.random.uniform(0, 0.1 * np.max(costs), size=len(feasible))
                noisy_costs = costs + noise
                best_f = feasible[np.argmin(noisy_costs)]
            
            result[i] = best_f
            load[best_f] += demands[i]
        
        if feasible_flag:
            # Repair just in case (though greedy should be feasible)
            result, load = repair(result, load)
            
            # Check if still feasible
            if np.all(load <= capacities):
                cost = compute_total_cost(result, load)
                if cost < best_init_cost:
                    best_init_cost = cost
                    best_init_assignment = result.copy()
                    best_init_loads = load.copy()
    
    # If no feasible solution found via greedy (should not happen), fallback to first feasible
    if best_init_assignment is None:
        result = np.zeros(n_customers, dtype=np.int64)
        load = np.zeros(n_facilities, dtype=np.int64)
        for i in range(n_customers):
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    break
        best_init_assignment = result
        best_init_loads = load
        best_init_cost = compute_total_cost(result, load)

    current_assignment = best_init_assignment
    current_loads = best_init_loads
    current_cost = best_init_cost

    # 2. Local Search
    # Precompute customer lists per facility for faster swap access
    def get_customer_lists(assignment):
        lists = [[] for _ in range(n_facilities)]
        for i in range(n_customers):
            lists[assignment[i]].append(i)
        return lists

    def try_improve(assignment, loads, cost):
        best_cost = cost
        best_assignment = assignment.copy()
        best_loads = loads.copy()
        improved = False
        
        # 1-Move
        for i in range(n_customers):
            cur_f = assignment[i]
            for f in range(n_facilities):
                if f == cur_f:
                    continue
                if loads[f] + demands[i] > capacities[f]:
                    continue
                
                new_loads = loads.copy()
                new_loads[cur_f] -= demands[i]
                new_loads[f] += demands[i]
                
                cost_change = assign_cost[i, f] - assign_cost[i, cur_f]
                
                if new_loads[cur_f] == 0:
                    cost_change -= open_cost[cur_f]
                if loads[f] == 0:
                    cost_change += open_cost[f]
                
                new_cost = best_cost + cost_change
                if new_cost < best_cost - 1e-9:
                    best_cost = new_cost
                    best_assignment = assignment.copy()
                    best_assignment[i] = f
                    best_loads = new_loads
                    improved = True
        
        # 2-Swap
        customer_lists = get_customer_lists(best_assignment)
        
        for f1 in range(n_facilities):
            if not customer_lists[f1]:
                continue
            for f2 in range(f1 + 1, n_facilities):
                if not customer_lists[f2]:
                    continue
                
                for i in customer_lists[f1]:
                    for j in customer_lists[f2]:
                        new_load_f1 = best_loads[f1] - demands[i] + demands[j]
                        new_load_f2 = best_loads[f2] - demands[j] + demands[i]
                        
                        if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                            continue
                        
                        cost_change = (assign_cost[i, f2] - assign_cost[i, f1]) + \
                                      (assign_cost[j, f1] - assign_cost[j, f2])
                        
                        f1_empty_after = (new_load_f1 == 0)
                        f2_empty_after = (new_load_f2 == 0)
                        f1_empty_before = (best_loads[f1] == 0)
                        f2_empty_before = (best_loads[f2] == 0)
                        
                        if f1_empty_after and not f1_empty_before:
                            cost_change -= open_cost[f1]
                        elif not f1_empty_after and f1_empty_before:
                            cost_change += open_cost[f1]
                        
                        if f2_empty_after and not f2_empty_before:
                            cost_change -= open_cost[f2]
                        elif not f2_empty_after and f2_empty_before:
                            cost_change += open_cost[f2]
                        
                        new_cost = best_cost + cost_change
                        if new_cost < best_cost - 1e-9:
                            best_cost = new_cost
                            best_assignment = best_assignment.copy()
                            best_assignment[i] = f2
                            best_assignment[j] = f1
                            best_loads = best_loads.copy()
                            best_loads[f1] = new_load_f1
                            best_loads[f2] = new_load_f2
                            improved = True
                            
        return best_assignment, best_loads, best_cost, improved

    max_iters = 100
    iters = 0
    while iters < max_iters:
        if time.time() > deadline - 1.0:
            break
            
        new_assignment, new_loads, new_cost, improved = try_improve(current_assignment, current_loads, current_cost)
        
        if not improved:
            break
            
        current_assignment = new_assignment
        current_loads = new_loads
        current_cost = new_cost
        iters += 1

    # Final check and repair
    current_assignment, current_loads = repair(current_assignment, current_loads)
    
    # Verify indices
    for i in range(n_customers):
        if current_assignment[i] < 0 or current_assignment[i] >= n_facilities:
             for f in range(n_facilities):
                 if current_loads[f] + demands[i] <= capacities[f]:
                     old_f = current_assignment[i]
                     if old_f >= 0 and old_f < n_facilities:
                         current_loads[old_f] -= demands[i]
                     current_assignment[i] = f
                     current_loads[f] += demands[i]
                     break

    return current_assignment
# EVOLVE-BLOCK-END

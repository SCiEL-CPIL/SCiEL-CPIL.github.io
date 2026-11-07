import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. Improved Greedy Initialization
    # Strategy: For each customer (sorted by demand desc), assign to facility
    # that minimizes (assign_cost + open_cost if facility is closed)
    # subject to capacity.
    
    order = np.argsort(-demands)
    assignment = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    open_mask = np.zeros(n_fac, dtype=bool)
    
    for i in order:
        best_f = -1
        best_cost_val = float('inf')
        
        d_i = demands[i]
        costs_i = assign_cost[i]
        
        for f in range(n_fac):
            if load[f] + d_i > capacities[f]:
                continue
            
            if open_mask[f]:
                c = costs_i[f]
            else:
                c = open_cost[f] + costs_i[f]
                
            if c < best_cost_val:
                best_cost_val = c
                best_f = f
                
        if best_f != -1:
            assignment[i] = best_f
            load[best_f] += d_i
            if not open_mask[best_f]:
                open_mask[best_f] = True
        else:
            # Fallback: find any feasible facility
            for f in range(n_fac):
                if load[f] + d_i <= capacities[f]:
                    assignment[i] = f
                    load[f] += d_i
                    if not open_mask[f]:
                        open_mask[f] = True
                    break
                    
    # If some customers are still unassigned (should not happen), assign to 0
    # This is a safety check
    unassigned = np.where(assignment == -1)[0]
    if len(unassigned) > 0:
         for i in unassigned:
             assignment[i] = 0
             load[0] += demands[i]
             if not open_mask[0]:
                 open_mask[0] = True

    # Helper to calculate cost
    def calc_cost(assignment, load):
        open_sum = np.sum(open_cost[load > 0])
        assign_sum = np.sum(assign_cost[np.arange(n_cust), assignment])
        return open_sum + assign_sum

    best_assignment = assignment.copy()
    best_load = load.copy()
    best_cost = calc_cost(best_assignment, best_load)
    
    # 2. Simulated Annealing Local Search
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    T = 100.0 # Initial temperature
    T_min = 0.1
    cooling = 0.999
    
    current_assignment = best_assignment.copy()
    current_load = best_load.copy()
    current_cost = best_cost
    
    # Precompute indices
    cust_indices = np.arange(n_cust)
    
    while time.time() < deadline:
        T *= cooling
        if T < T_min:
            T = T_min
            
        move_type = np.random.rand()
        
        if move_type < 0.5:
            # Move a customer
            i = np.random.randint(n_cust)
            f_old = current_assignment[i]
            f_new = np.random.randint(n_fac)
            
            if f_old == f_new:
                continue
                
            # Check capacity
            if current_load[f_new] + demands[i] > capacities[f_new]:
                continue
                
            # Calculate delta cost
            d_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
            
            d_open = 0.0
            # If f_old becomes empty, we save open_cost[f_old]
            if current_load[f_old] == demands[i]:
                d_open -= open_cost[f_old]
            # If f_new was empty, we pay open_cost[f_new]
            if current_load[f_new] == 0:
                d_open += open_cost[f_new]
                
            delta_cost = d_assign + d_open
            
            # Accept?
            if delta_cost < 0 or (T > 0 and np.random.rand() < np.exp(-delta_cost / T)):
                # Apply move
                current_assignment[i] = f_new
                current_load[f_old] -= demands[i]
                current_load[f_new] += demands[i]
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = current_assignment.copy()
                    best_load = current_load.copy()
                    
        elif move_type < 0.8:
            # Swap two customers
            i1 = np.random.randint(n_cust)
            i2 = np.random.randint(n_cust)
            if i1 == i2:
                continue
            
            f1 = current_assignment[i1]
            f2 = current_assignment[i2]
            
            if f1 == f2:
                continue
                
            # Check capacities after swap
            d1 = demands[i1]
            d2 = demands[i2]
            
            new_load_1 = current_load[f1] - d1 + d2
            new_load_2 = current_load[f2] - d2 + d1
            
            if new_load_1 > capacities[f1] or new_load_2 > capacities[f2]:
                continue
            
            # Calculate delta cost
            d_assign = (assign_cost[i1, f2] - assign_cost[i1, f1]) + \
                       (assign_cost[i2, f1] - assign_cost[i2, f2])
            
            d_open = 0.0
            old_open_1 = 1 if current_load[f1] > 0 else 0
            old_open_2 = 1 if current_load[f2] > 0 else 0
            
            new_open_1 = 1 if new_load_1 > 0 else 0
            new_open_2 = 1 if new_load_2 > 0 else 0
            
            d_open = (new_open_1 - old_open_1) * open_cost[f1] + \
                     (new_open_2 - old_open_2) * open_cost[f2]
            
            delta_cost = d_assign + d_open
            
            if delta_cost < 0 or (T > 0 and np.random.rand() < np.exp(-delta_cost / T)):
                # Apply swap
                current_assignment[i1] = f2
                current_assignment[i2] = f1
                current_load[f1] = new_load_1
                current_load[f2] = new_load_2
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = current_assignment.copy()
                    best_load = current_load.copy()

    return best_assignment

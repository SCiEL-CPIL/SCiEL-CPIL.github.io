import numpy as np
import time
import math

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Initial Solution: Greedy (Largest Demand First, Cheapest Feasible)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Fallback: should not happen if instance is feasible
            f_best = np.argmax(capacities - load)
            result[i] = f_best
            load[f_best] += demands[i]

    def get_cost(res, ld):
        total = 0.0
        for f in range(n_facilities):
            if ld[f] > 0:
                total += open_cost[f]
        total += np.sum(assign_cost[np.arange(n_customers), res])
        return total

    current_cost = get_cost(result, load)
    best_result = result.copy()
    best_load = load.copy()
    best_cost = current_cost
    
    start_time = time.time()
    time_limit = 15.0
    
    # Simulated Annealing Parameters
    avg_assign_cost = np.mean(assign_cost)
    avg_open_cost = np.mean(open_cost)
    T = max(avg_assign_cost * 10, avg_open_cost * 2)
    T_min = 1e-5
    alpha = 0.9995 # Slower cooling
    
    def try_move(i, f_new, res, ld):
        f_old = res[i]
        if f_new == f_old:
            return 0, False
        if ld[f_new] + demands[i] > capacities[f_new]:
            return 0, False
        
        delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
        delta_open = 0.0
        
        if ld[f_old] == demands[i]:
            delta_open -= open_cost[f_old]
            
        if ld[f_new] == 0:
            delta_open += open_cost[f_new]
            
        return delta_assign + delta_open, True

    def do_move(i, f_new, res, ld):
        f_old = res[i]
        ld[f_old] -= demands[i]
        ld[f_new] += demands[i]
        res[i] = f_new

    def try_swap(i, j, res, ld):
        f_i = res[i]
        f_j = res[j]
        if f_i == f_j:
            return 0, False
        
        if ld[f_i] - demands[i] + demands[j] > capacities[f_i]:
            return 0, False
        if ld[f_j] - demands[j] + demands[i] > capacities[f_j]:
            return 0, False
            
        delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
        
        return delta_assign, True

    def do_swap(i, j, res, ld):
        f_i = res[i]
        f_j = res[j]
        ld[f_i] -= demands[i]
        ld[f_j] -= demands[j]
        ld[f_i] += demands[j]
        ld[f_j] += demands[i]
        res[i] = f_j
        res[j] = f_i

    # Pre-compute indices
    cust_indices = np.arange(n_customers)
    
    # Batch processing to reduce time.time() overhead
    batch_size = 1000
    
    while time.time() - start_time < time_limit:
        for _ in range(batch_size):
            if time.time() - start_time >= time_limit:
                break
                
            move_type = np.random.random()
            
            if move_type < 0.6:
                # Try a move
                i = np.random.randint(n_customers)
                f_new = np.random.randint(n_facilities)
                delta, valid = try_move(i, f_new, result, load)
                
                if valid:
                    if delta < 0:
                        do_move(i, f_new, result, load)
                        current_cost += delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = result.copy()
                            best_load = load.copy()
                    else:
                        # Accept bad move with probability exp(-delta/T)
                        if T > 0:
                            # Optimization: if delta/T is very large, exp is 0
                            if -delta / T < 700: 
                                if np.random.random() < math.exp(-delta / T):
                                    do_move(i, f_new, result, load)
                                    current_cost += delta
            else:
                # Try a swap
                i = np.random.randint(n_customers)
                j = np.random.randint(n_customers)
                if i != j:
                    delta, valid = try_swap(i, j, result, load)
                    
                    if valid:
                        if delta < 0:
                            do_swap(i, j, result, load)
                            current_cost += delta
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_result = result.copy()
                                best_load = load.copy()
                        else:
                            if T > 0:
                                if -delta / T < 700:
                                    if np.random.random() < math.exp(-delta / T):
                                        do_swap(i, j, result, load)
                                        current_cost += delta
            
            T *= alpha
            if T < T_min:
                T = T_min

    return best_result

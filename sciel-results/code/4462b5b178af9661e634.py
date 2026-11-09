import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Initial Solution: Greedy (Largest Demand First, Cheapest Feasible)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        # Get facilities sorted by cost for this customer
        # We need to check capacity
        # To speed up, we can just iterate through sorted indices
        # Since n_facilities is small (20), sorting is cheap
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Fallback: should not happen if instance is feasible
            # Assign to facility with most remaining capacity
            f_best = np.argmax(capacities - load)
            result[i] = f_best
            load[f_best] += demands[i]

    def get_cost(res, ld):
        total = 0.0
        # Opening costs
        for f in range(n_facilities):
            if ld[f] > 0:
                total += open_cost[f]
        # Assignment costs
        total += np.sum(assign_cost[np.arange(n_customers), res])
        return total

    current_cost = get_cost(result, load)
    best_result = result.copy()
    best_load = load.copy()
    best_cost = current_cost

    start_time = time.time()
    time_limit = 15.0 # Leave buffer for 5 instances (24s each approx)
    
    def try_move(i, f_new, res, ld):
        f_old = res[i]
        if f_new == f_old:
            return 0, False
        # Check capacity
        if ld[f_new] + demands[i] > capacities[f_new]:
            return 0, False
        
        delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
        delta_open = 0.0
        
        # Check if f_old becomes empty
        if ld[f_old] == demands[i]:
            delta_open -= open_cost[f_old]
            
        # Check if f_new becomes open
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
        
        # Check capacities
        if ld[f_i] - demands[i] + demands[j] > capacities[f_i]:
            return 0, False
        if ld[f_j] - demands[j] + demands[i] > capacities[f_j]:
            return 0, False
            
        # As analyzed, swaps do not change open status of facilities
        # because both facilities remain non-empty (assuming demands > 0)
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

    iterations = 0
    while time.time() - start_time < time_limit:
        improved = True
        while improved:
            improved = False
            # 1. Best Improvement Move
            best_delta = 0
            best_i = -1
            best_f = -1
            for i in range(n_customers):
                f_old = result[i]
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    delta, valid = try_move(i, f_new, result, load)
                    if valid and delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_f = f_new
            
            if best_i != -1:
                do_move(best_i, best_f, result, load)
                current_cost += best_delta
                improved = True
                continue # Restart scan after move
                
            # 2. Best Improvement Swap
            best_delta_swap = 0
            best_i_swap = -1
            best_j_swap = -1
            # Optimize: Only check pairs in different facilities
            # Simple O(N^2) check
            for i in range(n_customers):
                f_i = result[i]
                for j in range(i + 1, n_customers):
                    f_j = result[j]
                    if f_i == f_j:
                        continue
                    delta, valid = try_swap(i, j, result, load)
                    if valid and delta < best_delta_swap:
                        best_delta_swap = delta
                        best_i_swap = i
                        best_j_swap = j
                        
            if best_i_swap != -1:
                do_swap(best_i_swap, best_j_swap, result, load)
                current_cost += best_delta_swap
                improved = True
                continue # Restart scan after swap

        # Update best if current is better
        if current_cost < best_cost:
            best_cost = current_cost
            best_result = result.copy()
            best_load = load.copy()
            
        # Perturbation
        # Random swap
        if n_customers >= 2:
            i = np.random.randint(n_customers)
            j = np.random.randint(n_customers)
            if i != j:
                # Check if swap is feasible (capacity wise)
                f_i = result[i]
                f_j = result[j]
                if f_i != f_j:
                    if load[f_i] - demands[i] + demands[j] <= capacities[f_i] and \
                       load[f_j] - demands[j] + demands[i] <= capacities[f_j]:
                        do_swap(i, j, result, load)
                        # Recalculate cost exactly to avoid drift
                        current_cost = get_cost(result, load)
                    else:
                        # Try a random move instead
                        i = np.random.randint(n_customers)
                        f_new = np.random.randint(n_facilities)
                        if f_new != result[i]:
                            delta, valid = try_move(i, f_new, result, load)
                            if valid:
                                do_move(i, f_new, result, load)
                                current_cost = get_cost(result, load)
                else:
                    # Same facility, do nothing or try move
                    i = np.random.randint(n_customers)
                    f_new = np.random.randint(n_facilities)
                    if f_new != result[i]:
                        delta, valid = try_move(i, f_new, result, load)
                        if valid:
                            do_move(i, f_new, result, load)
                            current_cost = get_cost(result, load)
        
        iterations += 1

    return best_result

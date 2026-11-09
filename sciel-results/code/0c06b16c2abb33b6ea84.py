import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Initial Solution: Randomized Greedy
    # We want a feasible start.
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand descending to handle large ones first (heuristic for feasibility)
    order = np.argsort(-demands)
    
    for i in order:
        # Find feasible facilities
        feasible = [f for f in range(n_facilities) if load[f] + demands[i] <= capacities[f]]
        if not feasible:
            # Fallback: Assign to facility with most space (might be infeasible if instance is bad, but assume feasible)
            f_best = np.argmax(capacities - load)
            feasible = [f_best]
        
        # Choose the one with minimum assignment cost among feasible
        # To add some randomness/variation for SA, we could pick from top k, but let's keep it greedy for valid start
        min_cost = np.inf
        best_f = feasible[0]
        for f in feasible:
            c = assign_cost[i, f]
            if c < min_cost:
                min_cost = c
                best_f = f
        
        result[i] = best_f
        load[best_f] += demands[i]

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

    # 2. Simulated Annealing
    start_time = time.time()
    time_limit = 15.0 # Leave buffer
    
    # SA Parameters
    T_initial = 1000.0
    T_final = 1.0
    # Estimate number of iterations we can do. 
    # Each iter is O(N) for move generation + O(1) acceptance. 
    # N=100. 15s is plenty. Let's aim for 50,000 iterations.
    n_iterations = 50000
    cooling_factor = (T_final / T_initial) ** (1.0 / n_iterations)
    
    T = T_initial
    rng = np.random.default_rng()

    for _ in range(n_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a random neighbor
        move_type = rng.random()
        
        if move_type < 0.5:
            # Move: Pick random customer, move to random facility
            i = rng.integers(0, n_customers)
            f_new = rng.integers(0, n_facilities)
            f_old = result[i]
            
            if f_new == f_old:
                continue
            
            # Check capacity
            if load[f_new] + demands[i] > capacities[f_new]:
                continue
                
            # Calculate Delta Cost
            delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
            delta_open = 0.0
            
            # f_old becomes empty?
            if load[f_old] == demands[i]:
                delta_open -= open_cost[f_old]
            # f_new becomes open?
            if load[f_new] == 0:
                delta_open += open_cost[f_new]
                
            delta_cost = delta_assign + delta_open
            
            # Acceptance
            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                # Apply Move
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                result[i] = f_new
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
                    best_load = load.copy()
                    
        else:
            # Swap: Pick two random customers, swap their facilities
            i = rng.integers(0, n_customers)
            j = rng.integers(0, n_customers)
            
            if i == j:
                continue
                
            f_i = result[i]
            f_j = result[j]
            
            if f_i == f_j:
                continue
            
            # Check capacities after swap
            # load[f_i] will change by -demands[i] + demands[j]
            # load[f_j] will change by -demands[j] + demands[i]
            if load[f_i] - demands[i] + demands[j] > capacities[f_i]:
                continue
            if load[f_j] - demands[j] + demands[i] > capacities[f_j]:
                continue
            
            # Calculate Delta Cost
            # Swaps don't change open status of f_i or f_j because they were both non-empty 
            # (since i was at f_i and j was at f_j, and demands > 0, load > 0).
            # Wait, if load[f_i] == demands[i] and we swap, f_i gets demands[j]. 
            # Is it possible demands[j] is 0? No, demands >= 5.
            # So f_i remains non-empty. f_j remains non-empty.
            # Thus delta_open = 0.
            
            delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
            delta_cost = delta_assign
            
            # Acceptance
            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                # Apply Swap
                load[f_i] -= demands[i]
                load[f_j] -= demands[j]
                load[f_i] += demands[j]
                load[f_j] += demands[i]
                result[i] = f_j
                result[j] = f_i
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
                    best_load = load.copy()

        # Cool down
        T *= cooling_factor

    # Final Local Search polish on the best found solution to ensure we are in a local optimum
    # This combines the global exploration of SA with the precision of LS.
    result = best_result.copy()
    load = best_load.copy()
    current_cost = best_cost
    
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

    # Simple First Improvement Hill Climbing on the best SA result
    for _ in range(100): # Limit polish steps
        improved = False
        for i in range(n_customers):
            f_old = result[i]
            for f_new in range(n_facilities):
                if f_new == f_old:
                    continue
                delta, valid = try_move(i, f_new, result, load)
                if valid and delta < -1e-9:
                    do_move(i, f_new, result, load)
                    current_cost += delta
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
            
    if current_cost < best_cost:
        best_cost = current_cost
        best_result = result.copy()

    return best_result

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    def compute_cost(result):
        """Compute total cost for an assignment."""
        cost = 0.0
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            f = int(result[i])
            cost += assign_cost[i, f]
            load[f] += demands[i]
        for f in range(n_facilities):
            if load[f] > 1e-9:
                cost += open_cost[f]
        return cost, load

    def greedy_construct(alpha=0.5):
        """
        Construct a solution using a Greedy Randomized approach.
        alpha controls the restrictiveness of the candidate list.
        """
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        
        # Sort customers by demand descending (harder to place first)
        # Or by min cost ascending (cheaper first)?
        # Let's try sorting by demand descending to handle capacity early.
        order = np.argsort(-demands)
        
        for i in order:
            # Calculate effective cost for each facility
            # Cost = assign_cost + open_cost if facility is empty
            costs = assign_cost[i, :].copy()
            for f in range(n_facilities):
                if load[f] == 0:
                    costs[f] += open_cost[f]
            
            # Filter out infeasible facilities
            feasible = [f for f in range(n_facilities) if load[f] + demands[i] <= capacities[f] + 1e-6]
            
            if not feasible:
                # Fallback: find facility with most remaining capacity
                # This should not happen if total capacity >= total demand
                remaining = capacities - load
                f = int(np.argmax(remaining))
                result[i] = f
                load[f] += demands[i]
                continue
            
            # Get costs for feasible facilities
            feasible_costs = np.array([costs[f] for f in feasible])
            
            # Determine RCL (Restricted Candidate List)
            min_cost = np.min(feasible_costs)
            max_cost = np.max(feasible_costs)
            threshold = min_cost + alpha * (max_cost - min_cost)
            
            candidates = [f for f in feasible if costs[f] <= threshold]
            
            if not candidates:
                candidates = feasible
            
            # Randomly select from candidates
            f = random.choice(candidates)
            result[i] = f
            load[f] += demands[i]
            
        return result, load

    def local_search(result, load, current_cost, time_limit):
        """Deterministic local search with 1-move and 2-swap."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1-move pass
            for i in range(n_customers):
                if time.time() > time_limit:
                    return result, current_cost, load
                old_f = result[i]
                best_delta = 0
                best_new_f = -1
                
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[i] > capacities[new_f] + 1e-6:
                        continue
                    
                    delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                    delta_open = 0.0
                    if load[old_f] == demands[i]:
                        delta_open -= open_cost[old_f]
                    if load[new_f] == 0:
                        delta_open += open_cost[new_f]
                    
                    delta_cost = delta_assign + delta_open
                    if delta_cost < best_delta - 1e-9:
                        best_delta = delta_cost
                        best_new_f = new_f
                
                if best_new_f != -1:
                    load[old_f] -= demands[i]
                    load[best_new_f] += demands[i]
                    result[i] = best_new_f
                    current_cost += best_delta
                    improved = True
            
            # 2-swap pass
            for i in range(n_customers):
                if time.time() > time_limit:
                    return result, current_cost, load
                old_f_i = result[i]
                for j in range(i + 1, n_customers):
                    if time.time() > time_limit:
                        return result, current_cost, load
                    old_f_j = result[j]
                    if old_f_i == old_f_j:
                        continue
                    
                    if load[old_f_j] - demands[j] + demands[i] <= capacities[old_f_j] + 1e-6 and \
                       load[old_f_i] - demands[i] + demands[j] <= capacities[old_f_i] + 1e-6:
                        
                        delta_assign = (assign_cost[i, old_f_j] - assign_cost[i, old_f_i]) + \
                                       (assign_cost[j, old_f_i] - assign_cost[j, old_f_j])
                        
                        delta_open = 0.0
                        new_load_i = load[old_f_i] - demands[i] + demands[j]
                        new_load_j = load[old_f_j] - demands[j] + demands[i]
                        
                        if new_load_i == 0 and load[old_f_i] != 0:
                            delta_open -= open_cost[old_f_i]
                        if new_load_j == 0 and load[old_f_j] != 0:
                            delta_open -= open_cost[old_f_j]
                        if new_load_i > 0 and load[old_f_i] == 0:
                            delta_open += open_cost[old_f_i]
                        if new_load_j > 0 and load[old_f_j] == 0:
                            delta_open += open_cost[old_f_j]
                        
                        delta_cost = delta_assign + delta_open
                        if delta_cost < -1e-9:
                            load[old_f_i] -= demands[i]
                            load[old_f_j] -= demands[j]
                            load[old_f_j] += demands[i]
                            load[old_f_i] += demands[j]
                            result[i] = old_f_j
                            result[j] = old_f_i
                            current_cost += delta_cost
                            improved = True
                            break
        
        return result, current_cost, load

    # Main Logic: GRASP
    best_result = None
    best_cost = np.inf
    best_load = None
    
    # Initial solution using a deterministic greedy (alpha=0) or LP if available?
    # Let's start with a few deterministic starts to get a good baseline
    for alpha in [0.0, 0.2, 0.5]:
        if time.time() > deadline - 10:
            break
        result, load = greedy_construct(alpha)
        cost, load = compute_cost(result)
        
        # Local Search
        time_limit_ls = min(deadline - 5, time.time() + 2)
        result, cost, load = local_search(result, load, cost, time_limit_ls)
        
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
            best_load = load.copy()
    
    # Iterated Local Search / GRASP iterations
    # Vary alpha to diversify
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    idx = 0
    
    while time.time() < deadline - 5:
        alpha = alphas[idx % len(alphas)]
        idx += 1
        
        # Construct
        result, load = greedy_construct(alpha)
        cost, load = compute_cost(result)
        
        # Local Search
        # Allocate time based on remaining time, but cap it to allow more iterations
        # If we have 50s left, spend 5s on LS, leaving 45s for 9 more iterations
        time_limit_ls = min(5.0, max(0.5, (deadline - 5 - time.time()) / 10)) 
        result, cost, load = local_search(result, load, cost, time_limit_ls)
        
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
            best_load = load.copy()
    
    # Final polish on the best found
    if best_result is not None and time.time() < deadline - 2:
        time_limit_final = deadline - 2
        best_result, best_cost, best_load = local_search(
            best_result, best_load, best_cost, time_limit_final
        )
    
    # Fallback if no valid result
    if best_result is None:
        best_result, best_load = greedy_construct(0.0)
        best_cost, _ = compute_cost(best_result)
        
    # Verify and fix if necessary
    load_check = np.zeros(n_facilities)
    for i in range(n_customers):
        f = int(best_result[i])
        if f < 0 or f >= n_facilities:
            best_result, best_load = greedy_construct(0.0)
            load_check = np.zeros(n_facilities)
            for i in range(n_customers):
                load_check[int(best_result[i])] += demands[i]
            break
        load_check[f] += demands[i]
    
    if np.any(load_check > capacities + 1e-6):
        # Force fix: move customers from overloaded facilities
        for f in range(n_facilities):
            if load_check[f] > capacities[f]:
                customers_in_f = [i for i in range(n_customers) if best_result[i] == f]
                for i in customers_in_f:
                    for new_f in range(n_facilities):
                        if new_f != f and load_check[new_f] + demands[i] <= capacities[new_f]:
                            load_check[f] -= demands[i]
                            load_check[new_f] += demands[i]
                            best_result[i] = new_f
                            break
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END

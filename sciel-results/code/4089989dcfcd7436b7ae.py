"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 130.0  # 150s limit, 20s safety margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Greedy seed: largest demand first, cheapest feasible facility
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    
    def total_cost(assignment):
        cost = 0.0
        for f in range(n_facilities):
            if np.any(assignment == f):
                cost += open_cost[f]
        cost += np.sum(assign_cost[np.arange(n_customers), assignment])
        return cost
    
    best_assignment = result.copy()
    best_cost = total_cost(best_assignment)
    
    # Local search: try moving each customer to a different facility
    def try_improve(assignment, max_iters=10000):
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            for i in range(n_customers):
                current_f = assignment[i]
                current_cost = assign_cost[i, current_f]
                
                # Try all other facilities
                for f in range(n_facilities):
                    if f == current_f:
                        continue
                    if load[f] + demands[i] > capacities[f]:
                        continue
                    
                    new_cost = assign_cost[i, f]
                    
                    # Estimate delta: change in assignment cost + change in open cost
                    delta_assign = new_cost - current_cost
                    delta_open = 0.0
                    
                    # If current facility becomes empty, we save its open cost
                    if load[current_f] == demands[i]:
                        # Check if any other customer is assigned to current_f
                        others = np.where((assignment == current_f) & (np.arange(n_customers) != i))[0]
                        if len(others) == 0:
                            delta_open -= open_cost[current_f]
                    
                    # If new facility was empty, we add its open cost
                    if load[f] == 0:
                        delta_open += open_cost[f]
                    
                    if delta_assign + delta_open < -1e-9:
                        # Accept the move
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        assignment[i] = f
                        improved = True
                        break
        
        return assignment
    
    # Multiple rounds of local search
    for round_num in range(5):
        if time.time() > deadline:
            break
        
        improved = try_improve(best_assignment, max_iters=5000)
        new_cost = total_cost(improved)
        
        if new_cost < best_cost - 1e-9:
            best_cost = new_cost
            best_assignment = improved
        else:
            break
    
    # Try a different initial ordering: smallest demand first
    if time.time() < deadline - 10:
        result2 = np.zeros(n_customers, dtype=int)
        load2 = np.zeros(n_facilities)
        for i in np.argsort(demands):
            for f in np.argsort(assign_cost[i]):
                if load2[f] + demands[i] <= capacities[f]:
                    load2[f] += demands[i]
                    result2[i] = f
                    break
        
        improved2 = try_improve(result2, max_iters=5000)
        cost2 = total_cost(improved2)
        
        if cost2 < best_cost - 1e-9:
            best_cost = cost2
            best_assignment = improved2
    
    # Try random restarts if time permits
    if time.time() < deadline - 20:
        rng = np.random.RandomState(42)
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            # Random assignment respecting capacities
            result_rand = np.zeros(n_customers, dtype=int)
            load_rand = np.zeros(n_facilities)
            
            # Shuffle customers
            order = rng.permutation(n_customers)
            for i in order:
                # Try facilities in random order
                fac_order = rng.permutation(n_facilities)
                assigned = False
                for f in fac_order:
                    if load_rand[f] + demands[i] <= capacities[f]:
                        load_rand[f] += demands[i]
                        result_rand[i] = f
                        assigned = True
                        break
                if not assigned:
                    # Fallback to cheapest feasible
                    for f in np.argsort(assign_cost[i]):
                        if load_rand[f] + demands[i] <= capacities[f]:
                            load_rand[f] += demands[i]
                            result_rand[i] = f
                            break
            
            improved_rand = try_improve(result_rand, max_iters=3000)
            cost_rand = total_cost(improved_rand)
            
            if cost_rand < best_cost - 1e-9:
                best_cost = cost_rand
                best_assignment = improved_rand
    
    return best_assignment
# EVOLVE-BLOCK-END

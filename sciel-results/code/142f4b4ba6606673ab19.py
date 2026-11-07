"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Deadline guard: leave 10s safety margin before the 120s limit
    deadline = time.time() + 110.0
    
    # 1. Greedy Initialization
    # Sort customers by demand descending to place large ones first (heuristic for feasibility)
    # Assign to cheapest feasible facility
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=np.float64)
    order = np.argsort(-demands)
    
    for i in order:
        # Find cheapest facility with remaining capacity
        # We can speed this up by sorting facilities by cost for this customer once
        # but n_fac is small (20), so simple loop is fine
        best_f = -1
        min_cost = np.inf
        for f in range(n_fac):
            if load[f] + demands[i] <= capacities[f]:
                if assign_cost[i, f] < min_cost:
                    min_cost = assign_cost[i, f]
                    best_f = f
        if best_f != -1:
            result[i] = best_f
            load[best_f] += demands[i]
        else:
            # Fallback if no feasible facility found (should not happen if total capacity > total demand)
            # Just pick any facility with capacity, ignoring cost
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    break
    
    # 2. Compute Initial Cost
    def compute_cost(res, load_arr):
        total_assign = np.sum(assign_cost[np.arange(n_cust), res])
        # Identify used facilities
        used_facs = np.unique(res)
        total_open = np.sum(open_cost[used_facs])
        return total_assign + total_open

    current_load = load.copy()
    current_cost = compute_cost(result, current_load)
    best_result = result.copy()
    best_cost = current_cost
    best_load = current_load.copy()

    # 3. Simulated Annealing
    # Parameters
    T_start = 1000.0
    T_end = 1.0
    alpha = 0.9995 # Cooling rate
    
    # Estimate number of iterations we can afford
    # Let's run until time runs out or T < T_end
    T = T_start
    
    # Precompute indices for faster access
    indices = np.arange(n_cust)
    
    while time.time() < deadline and T > T_end:
        # Pick a random neighborhood move
        # 50% Move, 50% Swap
        if np.random.rand() < 0.5:
            # Move: reassign customer i to facility f
            i = np.random.randint(n_cust)
            old_f = result[i]
            # Choose a random new facility f != old_f
            candidates = [f for f in range(n_fac) if f != old_f]
            if not candidates: continue
            new_f = candidates[np.random.randint(len(candidates))]
            
            # Check capacity
            if current_load[new_f] + demands[i] > capacities[new_f]:
                continue
            
            # Calculate Delta Cost
            # Assignment change
            delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
            
            # Opening cost change
            delta_open = 0.0
            # If old_f becomes empty
            if current_load[old_f] == demands[i]:
                delta_open -= open_cost[old_f]
            # If new_f was empty
            if current_load[new_f] == 0:
                delta_open += open_cost[new_f]
            
            delta_cost = delta_assign + delta_open
            
            # Acceptance
            if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
                # Apply move
                current_load[old_f] -= demands[i]
                current_load[new_f] += demands[i]
                result[i] = new_f
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
                    best_load = current_load.copy()
        
        else:
            # Swap: swap assignments of customer i and j
            i, j = np.random.choice(n_cust, 2, replace=False)
            f_i = result[i]
            f_j = result[j]
            
            if f_i == f_j:
                continue
            
            # Check capacity for swap
            # i moves from f_i to f_j, j moves from f_j to f_i
            # Load at f_i: -d_i + d_j
            # Load at f_j: -d_j + d_i
            
            new_load_f_i = current_load[f_i] - demands[i] + demands[j]
            new_load_f_j = current_load[f_j] - demands[j] + demands[i]
            
            if new_load_f_i < 0 or new_load_f_j < 0:
                continue
            if new_load_f_i > capacities[f_i] or new_load_f_j > capacities[f_j]:
                continue
            
            # Calculate Delta Cost
            # Assignment change
            delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + (assign_cost[j, f_i] - assign_cost[j, f_j])
            
            # Opening cost change
            delta_open = 0.0
            # Check if f_i becomes empty
            if current_load[f_i] == demands[i]:
                # It was only i, now it's j. If j has 0 demand? No, demands > 0.
                # So f_i is still used. No change in open cost for f_i?
                # Wait, if load[f_i] == d_i, then after swap load is d_j.
                # If d_j > 0, f_i is still open. So no change?
                # Unless d_j is 0, but demands are 5-35.
                # So opening status of f_i doesn't change if it was only i?
                # It was open (used by i). It will be open (used by j).
                # So delta_open for f_i is 0.
                pass
            
            # Check if f_j becomes empty
            if current_load[f_j] == demands[j]:
                # It was only j, now it's i. Still open.
                pass
            
            # Check if f_i was empty? No, it had i.
            # Check if f_j was empty? No, it had j.
            
            # Are there cases where opening status changes?
            # Only if a facility becomes empty or starts being used.
            # In a swap between two existing users, the set of used facilities usually stays the same.
            # Exceptions:
            # 1. f_i had ONLY i. After swap, it has j. Still used.
            # 2. f_j had ONLY j. After swap, it has i. Still used.
            # So delta_open is always 0 for a swap between two customers assigned to different facilities?
            # Yes, because both facilities remain in use (unless demand is 0, which it isn't).
            
            delta_cost = delta_assign
            
            # Acceptance
            if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
                # Apply swap
                current_load[f_i] = new_load_f_i
                current_load[f_j] = new_load_f_j
                result[i] = f_j
                result[j] = f_i
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
                    best_load = current_load.copy()
        
        T *= alpha

    return best_result
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 135.0
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    N = n_customers
    F = n_facilities
    
    def greedy_fallback():
        result = np.zeros(N, dtype=int)
        load = np.zeros(F)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    result[i] = f
                    break
        return result, load
    
    # Try LP formulation: min-cost flow with fixed charge via big-M
    # Variables: x[i, f] = 1 if customer i assigned to facility f
    # This is actually a set partitioning problem, not a flow.
    # Let's use a different approach: min-cost flow where each customer sends demand to a facility.
    
    # Actually, the problem is: assign each customer to exactly one facility, capacity constraints, fixed costs.
    # This is a mixed integer program. Let's try a different structural approach.
    
    # Approach: Use scipy.optimize.milp if available, otherwise fall back to heuristic.
    # But milp might be slow. Let's try a smarter heuristic.
    
    # New structural approach: 
    # 1. Start with all facilities open, assign each customer to nearest facility
    # 2. Iteratively close facilities that are expensive to keep open relative to reassignment cost
    # 3. Use local search with 1-move, 2-swap, and 3-exchange moves
    
    # First, get a feasible solution via greedy
    best_assignment, best_load = greedy_fallback()
    
    def calc_cost(assignment):
        total_assign = np.sum(assign_cost[np.arange(N), assignment])
        used = np.unique(assignment)
        total_open = np.sum(open_cost[used])
        return total_assign + total_open
    
    def is_valid(assignment):
        load = np.zeros(F)
        for i in range(N):
            load[assignment[i]] += demands[i]
        return np.all(load <= capacities + 1e-6)
    
    if not is_valid(best_assignment):
        # Try a different greedy: sort by demand descending, assign to facility with most remaining capacity
        result = np.zeros(N, dtype=int)
        load = np.zeros(F)
        for i in np.argsort(-demands):
            # Find facility with most remaining capacity that can accommodate
            best_f = -1
            best_rem = -1
            for f in range(F):
                rem = capacities[f] - load[f]
                if rem >= demands[i] and rem > best_rem:
                    best_rem = rem
                    best_f = f
            if best_f == -1:
                # Any feasible facility
                for f in range(F):
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        best_f = f
                        break
            if best_f != -1:
                load[best_f] += demands[i]
                result[i] = best_f
            else:
                # Should not happen if total capacity >= total demand
                result[i] = 0
        best_assignment = result
        best_load = load.copy()
    
    best_cost = calc_cost(best_assignment)
    load = np.zeros(F)
    for i in range(N):
        load[best_assignment[i]] += demands[i]
    
    # Local search with multiple move types
    improved = True
    max_iterations = 500
    iteration = 0
    
    while improved and time.time() < deadline - 10.0 and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Pass 1: Single moves - move one customer to a different facility
        for i in range(N):
            if time.time() > deadline - 10.0:
                break
            current_f = best_assignment[i]
            # Try moving to each other facility
            for f in range(F):
                if f == current_f:
                    continue
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    new_cost = best_cost - assign_cost[i, current_f] + assign_cost[i, f]
                    if np.sum(best_assignment == current_f) == 1:
                        new_cost -= open_cost[current_f]
                    if np.sum(best_assignment == f) == 0:
                        new_cost += open_cost[f]
                    if new_cost < best_cost - 1e-9:
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        best_assignment[i] = f
                        best_cost = new_cost
                        improved = True
                        break  # Move to next customer
        
        # Pass 2: Pair swaps - swap two customers between different facilities
        if time.time() > deadline - 10.0:
            break
        for i in range(N):
            if time.time() > deadline - 10.0:
                break
            fi = best_assignment[i]
            for j in range(i + 1, N):
                if time.time() > deadline - 10.0:
                    break
                fj = best_assignment[j]
                if fi == fj:
                    continue
                # Check capacity feasibility for swap
                if (load[fi] - demands[i] + demands[j] <= capacities[fi] + 1e-9 and
                    load[fj] - demands[j] + demands[i] <= capacities[fj] + 1e-9):
                    new_cost = (best_cost - assign_cost[i, fi] - assign_cost[j, fj] +
                                assign_cost[i, fj] + assign_cost[j, fi])
                    # Open cost doesn't change in a swap between two open facilities
                    if new_cost < best_cost - 1e-9:
                        load[fi] = load[fi] - demands[i] + demands[j]
                        load[fj] = load[fj] - demands[j] + demands[i]
                        best_assignment[i] = fj
                        best_assignment[j] = fi
                        best_cost = new_cost
                        improved = True
        
        # Pass 3: Try closing a facility and reassigning all its customers
        if time.time() > deadline - 10.0:
            break
        used_facilities = np.unique(best_assignment)
        for f in used_facilities.copy():
            if time.time() > deadline - 10.0:
                break
            custs_f = np.where(best_assignment == f)[0]
            if len(custs_f) == 0:
                continue
            
            # Try to reassign all customers at f to other facilities
            new_load = load.copy()
            new_load[f] = 0
            new_assign = best_assignment.copy()
            feasible = True
            new_assign_cost = 0.0
            
            # Sort customers by demand descending for better packing
            sorted_custs = custs_f[np.argsort(-demands[custs_f])]
            
            for i in sorted_custs:
                best_k = -1
                best_c = np.inf
                for k in range(F):
                    if k == f:
                        continue
                    if new_load[k] + demands[i] <= capacities[k] + 1e-9:
                        c = assign_cost[i, k]
                        if c < best_c:
                            best_c = c
                            best_k = k
                if best_k == -1:
                    feasible = False
                    break
                new_assign[i] = best_k
                new_load[best_k] += demands[i]
                new_assign_cost += assign_cost[i, best_k]
            
            if feasible:
                # Calculate new total cost
                new_used = np.unique(new_assign)
                new_open_cost = np.sum(open_cost[new_used])
                total_new_assign = 0.0
                for i in range(N):
                    total_new_assign += assign_cost[i, new_assign[i]]
                new_total = total_new_assign + new_open_cost
                
                if new_total < best_cost - 1e-9:
                    best_assignment = new_assign
                    load = new_load
                    best_cost = new_total
                    improved = True
    
    # Final verification
    if not is_valid(best_assignment):
        fallback, _ = greedy_fallback()
        return fallback
    
    return best_assignment
# EVOLVE-BLOCK-END

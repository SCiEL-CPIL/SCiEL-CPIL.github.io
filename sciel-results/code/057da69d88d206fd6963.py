import numpy as np
import time
import random

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Calculate total cost of a solution."""
    n_cust = len(result)
    n_fac = len(open_cost)
    
    # Check validity
    for i in range(n_cust):
        if result[i] < 0 or result[i] >= n_fac:
            return float('inf')
            
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        
    if np.any(load > capacities + 1e-6):
        return float('inf')
        
    total_assign = np.sum(assign_cost[np.arange(n_cust), result])
    
    used = np.zeros(n_fac, dtype=bool)
    for f in result:
        used[f] = True
        
    total_open = np.sum(open_cost[used])
    
    return total_assign + total_open

def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """
    Best-Improvement Local Search with Moves and Swaps.
    Returns improved result and cost.
    """
    n_cust = len(result)
    n_fac = len(open_cost)
    
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        
    # Initialize cost
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if current_cost == float('inf'):
        return result, current_cost
        
    # Precompute customer counts per facility to quickly check if a facility becomes empty
    fac_counts = np.zeros(n_fac, dtype=int)
    for f in result:
        fac_counts[f] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # --- Phase 1: Single Moves ---
        best_delta = 1e-9
        best_move = None # (i, from_f, to_f)
        
        for i in range(n_cust):
            from_f = result[i]
            # Check all other facilities
            for to_f in range(n_fac):
                if to_f == from_f:
                    continue
                # Capacity check
                if load[to_f] + demands[i] > capacities[to_f] + 1e-6:
                    continue
                
                # Calculate delta
                delta_assign = assign_cost[i, to_f] - assign_cost[i, from_f]
                
                delta_open = 0.0
                # If from_f becomes empty, save open_cost[from_f]
                if fac_counts[from_f] == 1:
                    delta_open -= open_cost[from_f]
                # If to_f was empty, pay open_cost[to_f]
                if fac_counts[to_f] == 0:
                    delta_open += open_cost[to_f]
                    
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_move = (i, from_f, to_f)
                    
        if best_move is not None:
            i, from_f, to_f = best_move
            # Apply move
            load[from_f] -= demands[i]
            load[to_f] += demands[i]
            fac_counts[from_f] -= 1
            fac_counts[to_f] += 1
            result[i] = to_f
            current_cost += best_delta
            improved = True
            continue # Restart search with new state
            
        # --- Phase 2: Pairwise Swaps ---
        # Only check swaps between customers in different facilities
        # To optimize, we can iterate over facilities
        best_delta = 1e-9
        best_swap = None # (i, j, f_i, f_j)
        
        # Group customers by facility for faster iteration
        fac_to_custs = [[] for _ in range(n_fac)]
        for i in range(n_cust):
            fac_to_custs[result[i]].append(i)
            
        for f1 in range(n_fac):
            if not fac_to_custs[f1]:
                continue
            for f2 in range(f1 + 1, n_fac):
                if not fac_to_custs[f2]:
                    continue
                
                for i in fac_to_custs[f1]:
                    for j in fac_to_custs[f2]:
                        # Capacity check after swap
                        # f1 loses i, gains j. f2 loses j, gains i.
                        if load[f1] - demands[i] + demands[j] > capacities[f1] + 1e-6:
                            continue
                        if load[f2] - demands[j] + demands[i] > capacities[f2] + 1e-6:
                            continue
                        
                        delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - \
                                       (assign_cost[i, f1] + assign_cost[j, f2])
                        
                        delta_open = 0.0
                        # Swaps don't change the set of used facilities unless one becomes empty?
                        # No, both facilities remain used because they each lose one and gain one.
                        # So delta_open is always 0 for swaps between non-empty facilities.
                        # Wait, if fac_counts[f1] == 1, then after swap it has 1 (j). So it stays open.
                        # If fac_counts[f2] == 1, then after swap it has 1 (i). So it stays open.
                        # So yes, delta_open is 0.
                        
                        total_delta = delta_assign
                        
                        if total_delta < best_delta:
                            best_delta = total_delta
                            best_swap = (i, j, f1, f2)
                            
        if best_swap is not None:
            i, j, f1, f2 = best_swap
            # Apply swap
            load[f1] += demands[j] - demands[i]
            load[f2] += demands[i] - demands[j]
            result[i] = f2
            result[j] = f1
            # fac_counts don't change for swaps
            current_cost += best_delta
            improved = True
            continue # Restart search

    return result, current_cost

def _perturb(result, demands, capacities, open_cost, assign_cost, strength=5):
    """
    Perturb the solution by making 'strength' random feasible moves.
    """
    n_cust = len(result)
    n_fac = len(open_cost)
    
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        
    for _ in range(strength):
        # Pick a random customer
        i = np.random.randint(n_cust)
        from_f = result[i]
        
        # Find feasible facilities
        feasible = []
        for f in range(n_fac):
            if f != from_f and load[f] + demands[i] <= capacities[f] + 1e-6:
                feasible.append(f)
                
        if not feasible:
            continue
            
        to_f = feasible[np.random.randint(len(feasible))]
        
        # Apply move
        load[from_f] -= demands[i]
        load[to_f] += demands[i]
        result[i] = to_f
        
    return result

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(open_cost)
    
    # 1. Initialization: Greedy
    # Sort customers by demand descending to place large ones first
    order = np.argsort(-demands)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    for i in order:
        # Find cheapest feasible facility
        best_f = -1
        best_cost = float('inf')
        for f in range(n_fac):
            if load[f] + demands[i] <= capacities[f] + 1e-6:
                cost = assign_cost[i, f]
                # Tie-breaker: prefer facilities already open? 
                # Or just cheapest assignment. Local search will handle opening costs.
                if cost < best_cost:
                    best_cost = cost
                    best_f = f
        
        if best_f == -1:
            # Fallback: facility with most remaining capacity
            remaining = capacities - load
            best_f = np.argmax(remaining)
            
        result[i] = best_f
        load[best_f] += demands[i]
        
    # Ensure no -1 left (should not happen if capacities are sufficient, but just in case)
    for i in range(n_cust):
        if result[i] == -1:
             result[i] = 0

    # 2. Local Search
    best_result, best_cost = _local_search(result, demands, capacities, open_cost, assign_cost, deadline)
    
    # 3. Iterated Local Search
    # If time remains, perform ILS
    while time.time() < deadline - 10.0: # Keep 10s buffer
        # Perturb
        perturbed = _perturb(best_result, demands, capacities, open_cost, assign_cost, strength=3)
        
        # Local Search on perturbed
        new_result, new_cost = _local_search(perturbed, demands, capacities, open_cost, assign_cost, deadline)
        
        if new_cost < best_cost:
            best_result = new_result
            best_cost = new_cost
            
        # If we are running out of time, break
        if time.time() > deadline - 5.0:
            break
            
    # Final Verification
    load = np.zeros(n_fac)
    for i in range(n_cust):
        if best_result[i] < 0 or best_result[i] >= n_fac:
            best_result[i] = 0
        load[best_result[i]] += demands[i]
        
    if np.any(load > capacities + 1e-6):
        # Fallback to greedy if infeasible (should not happen with valid LS)
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_cost = float('inf')
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f] + 1e-6:
                    cost = assign_cost[i, f]
                    if cost < best_cost:
                        best_cost = cost
                        best_f = f
            if best_f == -1:
                remaining = capacities - load
                best_f = np.argmax(remaining)
            result[i] = best_f
            load[best_f] += demands[i]
        best_result = result

    return best_result.astype(int)

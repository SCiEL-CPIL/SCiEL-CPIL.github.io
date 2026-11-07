"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute the objective value for a given assignment. Returns inf if invalid."""
    n_fac = len(capacities)
    n_cust = len(demands)
    if len(result) != n_cust:
        return float('inf')
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return float('inf')
        load[f] += demands[i]
    if np.any(load > capacities + 1e-9):
        return float('inf')
    open_facilities = np.unique(result)
    cost = np.sum(open_cost[open_facilities]) + np.sum(assign_cost[np.arange(n_cust), result])
    return cost

def _greedy_assign(demands, capacities, open_cost, assign_cost):
    """Largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        # Sort facilities by cost for this customer
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Fallback: pick facility with most remaining capacity
            remaining = capacities - load
            f = np.argmax(remaining)
            if remaining[f] >= demands[i]:
                load[f] += demands[i]
                result[i] = f
            else:
                result[i] = 0
    return result

def _greedy_assign_smallest(demands, capacities, open_cost, assign_cost):
    """Smallest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(demands)
    for i in order:
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            remaining = capacities - load
            f = np.argmax(remaining)
            if remaining[f] >= demands[i]:
                load[f] += demands[i]
                result[i] = f
            else:
                result[i] = 0
    return result

def _fast_local_search_polish(result, demands, capacities, open_cost, assign_cost, time_limit=5.0):
    """
    Fast local search with incremental cost updates.
    Tries single reassignments. Used as a final polish.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    current = result.astype(int).copy()
    load = np.zeros(n_fac)
    for i, f in enumerate(current):
        load[f] += demands[i]

    fac_count = np.zeros(n_fac, dtype=int)
    for f in current:
        fac_count[f] += 1

    def get_open_cost():
        return np.sum(open_cost[fac_count > 0])

    def get_assign_cost():
        return np.sum(assign_cost[np.arange(n_cust), current])

    current_cost = get_open_cost() + get_assign_cost()
    best_cost = current_cost
    best_result = current.copy()
    start = time.time()

    improved = True
    while improved and (time.time() - start < time_limit):
        improved = False
        for i in range(n_cust):
            if time.time() - start > time_limit:
                break
            cur_f = current[i]
            d_i = demands[i]
            # Try moving to each other facility
            for new_f in range(n_fac):
                if new_f == cur_f:
                    continue
                if load[new_f] + d_i > capacities[new_f]:
                    continue

                delta = 0.0
                delta += assign_cost[i, new_f] - assign_cost[i, cur_f]

                new_cur_count = fac_count[cur_f] - 1
                new_new_count = fac_count[new_f] + 1

                if new_cur_count == 0 and fac_count[cur_f] > 0:
                    delta -= open_cost[cur_f]
                if new_new_count == 1 and fac_count[new_f] == 0:
                    delta += open_cost[new_f]

                if delta < -1e-9:
                    load[cur_f] -= d_i
                    load[new_f] += d_i
                    fac_count[cur_f] -= 1
                    fac_count[new_f] += 1
                    current[i] = new_f
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()
                    break  # restart inner loop for this customer

    return best_result

def _simulated_annealing(result, demands, capacities, open_cost, assign_cost, time_limit=100.0):
    """
    Simulated Annealing to optimize the assignment.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Convert to lists/arrays for fast access
    demands = np.array(demands)
    capacities = np.array(capacities)
    open_cost = np.array(open_cost)
    assign_cost = np.array(assign_cost)
    
    current = result.astype(int).copy()
    load = np.zeros(n_fac)
    fac_count = np.zeros(n_fac, dtype=int)
    for i, f in enumerate(current):
        load[f] += demands[i]
        fac_count[f] += 1

    def calc_cost():
        open_cost_val = np.sum(open_cost[fac_count > 0])
        assign_cost_val = np.sum(assign_cost[np.arange(n_cust), current])
        return open_cost_val + assign_cost_val

    current_cost = calc_cost()
    best_cost = current_cost
    best_result = current.copy()
    
    # Initial temperature based on average assignment cost
    avg_assign_cost = np.mean(assign_cost)
    T = avg_assign_cost * 2.0
    T_min = 1e-4
    cooling_rate = 0.999 # Tune this
    
    start_time = time.time()
    
    # To speed up random selection, we can pre-generate random indices? 
    # No, just use random module.
    
    while T > T_min and (time.time() - start_time < time_limit):
        # Decide move type: 90% single reassign, 10% swap
        if random.random() < 0.9:
            # Single reassign
            i = random.randint(0, n_cust - 1)
            cur_f = current[i]
            d_i = demands[i]
            
            # Pick a random new facility
            new_f = random.randint(0, n_fac - 1)
            if new_f == cur_f:
                continue
            
            if load[new_f] + d_i > capacities[new_f]:
                continue
            
            # Calculate delta
            delta = 0.0
            delta += assign_cost[i, new_f] - assign_cost[i, cur_f]
            
            # Open cost change
            if fac_count[cur_f] == 1:
                delta -= open_cost[cur_f]
            if fac_count[new_f] == 0:
                delta += open_cost[new_f]
            
            # Acceptance criterion
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Apply move
                load[cur_f] -= d_i
                load[new_f] += d_i
                fac_count[cur_f] -= 1
                fac_count[new_f] += 1
                current[i] = new_f
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = current.copy()
        else:
            # Swap move
            # Pick two distinct customers
            i = random.randint(0, n_cust - 1)
            j = random.randint(0, n_cust - 1)
            if i == j:
                continue
            
            f_i = current[i]
            f_j = current[j]
            if f_i == f_j:
                continue
            
            d_i = demands[i]
            d_j = demands[j]
            
            # Check capacity after swap
            # f_i loses i, gains j
            if load[f_i] - d_i + d_j > capacities[f_i]:
                continue
            # f_j loses j, gains i
            if load[f_j] - d_j + d_i > capacities[f_j]:
                continue
            
            # Calculate delta
            delta = 0.0
            delta += assign_cost[i, f_j] + assign_cost[j, f_i] - assign_cost[i, f_i] - assign_cost[j, f_j]
            
            # Open cost change for swap between two different facilities
            # fac_count[f_i] stays same (loses 1, gains 1)
            # fac_count[f_j] stays same (loses 1, gains 1)
            # So delta open cost is 0
            
            # Acceptance criterion
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Apply swap
                load[f_i] = load[f_i] - d_i + d_j
                load[f_j] = load[f_j] - d_j + d_i
                current[i] = f_j
                current[j] = f_i
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = current.copy()
        
        # Cool down
        T *= cooling_rate
        
        # Adaptive cooling: if no improvement for a long time, cool faster?
        # For now, fixed cooling rate is simpler and robust.
        
    return best_result

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. Generate multiple initial solutions
    candidates = []
    
    # Greedy: Largest demand first
    r1 = _greedy_assign(demands, capacities, open_cost, assign_cost)
    candidates.append(r1)
    
    # Greedy: Smallest demand first
    r2 = _greedy_assign_smallest(demands, capacities, open_cost, assign_cost)
    candidates.append(r2)
    
    # Greedy: Randomized (with some randomness in order)
    r3 = np.zeros(n_cust, dtype=int)
    load3 = np.zeros(n_fac)
    order3 = np.random.permutation(n_cust) # Random order
    for i in order3:
        fac_order = np.argsort(assign_cost[i])
        # Add some randomness to facility selection if multiple are close?
        # For now, just pick best feasible
        for f in fac_order:
            if load3[f] + demands[i] <= capacities[f]:
                load3[f] += demands[i]
                r3[i] = f
                break
        else:
            remaining = capacities - load3
            f = np.argmax(remaining)
            if remaining[f] >= demands[i]:
                load3[f] += demands[i]
                r3[i] = f
            else:
                r3[i] = 0
    candidates.append(r3)
    
    # Evaluate candidates and pick the best
    best_init = None
    best_init_cost = float('inf')
    for c in candidates:
        cost = _compute_cost(c, demands, capacities, open_cost, assign_cost)
        if cost < best_init_cost:
            best_init_cost = cost
            best_init = c
    
    if best_init is None:
        return np.zeros(n_cust, dtype=int) # Fallback
        
    # 2. Simulated Annealing
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        sa_result = _simulated_annealing(best_init, demands, capacities, open_cost, assign_cost, 
                                         time_limit=remaining_time - 5)
    else:
        sa_result = best_init
        
    # 3. Final Polish with Local Search
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        final_result = _fast_local_search_polish(sa_result, demands, capacities, open_cost, assign_cost, 
                                                 time_limit=remaining_time - 1)
    else:
        final_result = sa_result
        
    # 4. Validation
    final_cost = _compute_cost(final_result, demands, capacities, open_cost, assign_cost)
    if final_cost == float('inf'):
        # Fallback to the best initialization if SA/LS broke it
        final_result = best_init
        
    return final_result
# EVOLVE-BLOCK-END

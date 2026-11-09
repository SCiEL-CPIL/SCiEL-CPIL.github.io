"""Capacitated facility location: Vectorized local search with swaps and facility closing."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _calc_cost(demands, capacities, open_cost, assign_cost, assign):
    """Calculate total cost: open costs + assignment costs."""
    n = len(demands)
    assign = np.asarray(assign)
    # Assignment cost
    assign_sum = np.sum(assign_cost[np.arange(n), assign])
    # Open cost
    used = np.unique(assign)
    if len(used) > 0:
        open_sum = np.sum(open_cost[used])
    else:
        open_sum = 0.0
    return open_sum + assign_sum

def _local_search(demands, capacities, open_cost, assign_cost, assign, deadline, max_iters=1000):
    """
    Vectorized local search with single moves and swap moves.
    Returns improved assign array and cost.
    """
    n = len(demands)
    m = len(capacities)
    assign = assign.copy().astype(int)
    load = np.zeros(m, dtype=float)
    for i in range(n):
        load[assign[i]] += demands[i]
    
    current_cost = _calc_cost(demands, capacities, open_cost, assign_cost, assign)
    used = np.unique(assign)
    
    # Precompute current assignment costs
    curr_assign_costs = assign_cost[np.arange(n), assign]
    
    improved = True
    iter_count = 0
    
    while improved and time.time() < deadline - 2.0 and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # --- Single Move Pass (Vectorized) ---
        # We want to find for each customer i, the best facility f != assign[i]
        # such that load[f] + demands[i] <= capacities[f]
        # Delta = assign_cost[i, f] - assign_cost[i, assign[i]] + open_delta
        
        # To vectorize efficiently, we can check all facilities for all customers.
        # This is O(n*m) which is 100*20 = 2000 ops, very fast.
        
        # 1. Calculate base deltas for assignment cost
        # base_delta[i, f] = assign_cost[i, f] - curr_assign_costs[i]
        base_deltas = assign_cost - curr_assign_costs[:, np.newaxis]
        
        # 2. Apply capacity constraints
        # feasible[i, f] = (load[f] + demands[i] <= capacities[f])
        # Note: We must also exclude current facility f == assign[i]
        
        # Create mask for capacity feasibility
        # demands shape (n,), load shape (m,)
        # feasible_cap[i, f] = load[f] + demands[i] <= capacities[f]
        feasible_cap = (load[np.newaxis, :] + demands[:, np.newaxis]) <= capacities[np.newaxis, :]
        
        # Mask out current facility
        curr_fac = assign[:, np.newaxis]
        is_current = (curr_fac == np.arange(m)[np.newaxis, :])
        feasible = feasible_cap & (~is_current)
        
        # 3. Calculate Open Cost Deltas
        # If we move i from f_old to f_new:
        # - If load[f_old] becomes 0, we save open_cost[f_old]
        # - If load[f_new] was 0, we pay open_cost[f_new]
        
        # This is tricky to vectorize perfectly because "load becomes 0" depends on the specific move.
        # However, for a first pass, we can approximate or handle it in a second step.
        # Let's do a simplified vectorized move first, ignoring open cost changes for the delta calculation,
        # then verify. Or better, compute the potential delta including open costs.
        
        # Let's compute the delta array with open costs.
        # open_delta[i, f] = 
        #   + open_cost[f] if load[f] == 0 and f != assign[i]
        #   - open_cost[assign[i]] if load[assign[i]] == demands[i] (i.e., i is the last customer)
        #   0 otherwise
        
        # Note: The -open_cost part is constant for all f for a given i, so it doesn't affect argmin.
        # The +open_cost part depends on f.
        
        # Add open cost penalty for opening new facilities
        # If load[f] == 0, moving to f incurs open_cost[f]
        open_penalty = np.where(load[np.newaxis, :] == 0, open_cost[np.newaxis, :], 0.0)
        
        # We don't subtract the old open cost in the delta matrix because it's constant for row i.
        # However, we must ensure we don't "open" a facility that is already open.
        # The condition load[f] == 0 handles this: if load[f] > 0, penalty is 0.
        
        total_deltas = base_deltas + open_penalty
        
        # Apply feasibility mask
        total_deltas[~feasible] = np.inf
        
        # Find best move for each customer
        best_deltas = np.min(total_deltas, axis=1)
        best_facs = np.argmin(total_deltas, axis=1)
        
        # Find customers that can improve
        improve_mask = best_deltas < -1e-9
        if not np.any(improve_mask):
            # No single move improves, try swaps or break
            pass
        else:
            # Apply moves one by one to ensure capacity constraints are respected dynamically
            # Since moves are small, doing them sequentially is safer and still fast for n=100
            for i in np.where(improve_mask)[0]:
                f_new = best_facs[i]
                f_old = assign[i]
                
                # Re-check capacity (in case previous moves in this loop changed load)
                if load[f_new] + demands[i] <= capacities[f_new]:
                    # Perform move
                    load[f_old] -= demands[i]
                    load[f_new] += demands[i]
                    assign[i] = f_new
                    curr_assign_costs[i] = assign_cost[i, f_new]
                    current_cost += best_deltas[i] # Approximate, will recalc at end
                    improved = True
                    
        if improved:
            continue # Restart loop to recalculate deltas based on new loads
            
        # --- Swap Move Pass ---
        # If no single move improves, try swaps.
        # Sample random pairs (i, j)
        # If i and j are in different facilities f_i, f_j
        # Check if swapping is feasible and improves cost.
        
        found_swap = False
        # Number of samples depends on time remaining
        samples_left = int((deadline - time.time()) / 0.001) # ~1ms per sample is generous
        samples = min(5000, samples_left)
        
        for _ in range(samples):
            if time.time() > deadline - 2.0:
                break
            i, j = np.random.randint(0, n, size=2)
            if i == j:
                continue
            f_i = assign[i]
            f_j = assign[j]
            if f_i == f_j:
                continue
            
            d_i = demands[i]
            d_j = demands[j]
            
            # Check capacity feasibility for swap
            # New load f_i = load[f_i] - d_i + d_j
            # New load f_j = load[f_j] - d_j + d_i
            if load[f_i] - d_i + d_j > capacities[f_i]:
                continue
            if load[f_j] - d_j + d_i > capacities[f_j]:
                continue
            
            # Calculate cost delta
            # Assign cost delta: (cost[i, f_j] + cost[j, f_i]) - (cost[i, f_i] + cost[j, f_j])
            delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
            
            # Open cost delta
            # This is complex. 
            # Old open: open_cost[f_i] if load[f_i]>0, open_cost[f_j] if load[f_j]>0
            # New open: open_cost[f_i] if (load[f_i]-d_i+d_j)>0, open_cost[f_j] if (load[f_j]-d_j+d_i)>0
            
            old_open_i = open_cost[f_i] if load[f_i] > 0 else 0.0
            old_open_j = open_cost[f_j] if load[f_j] > 0 else 0.0
            
            new_load_i = load[f_i] - d_i + d_j
            new_load_j = load[f_j] - d_j + d_i
            
            new_open_i = open_cost[f_i] if new_load_i > 0 else 0.0
            new_open_j = open_cost[f_j] if new_load_j > 0 else 0.0
            
            delta_open = (new_open_i + new_open_j) - (old_open_i + old_open_j)
            
            total_delta = delta_assign + delta_open
            
            if total_delta < -1e-9:
                # Perform swap
                assign[i], assign[j] = assign[j], assign[i]
                load[f_i] = new_load_i
                load[f_j] = new_load_j
                curr_assign_costs[i] = assign_cost[i, f_j]
                curr_assign_costs[j] = assign_cost[j, f_i]
                current_cost += total_delta
                improved = True
                found_swap = True
                break # Restart loop for new state
                
        if found_swap:
            continue
            
        # --- Facility Closing Pass ---
        # Try to close a facility.
        # For each open facility f, check if we can reassign all its customers to others.
        # If savings in open_cost > increase in assign_cost, do it.
        
        used_facilities = np.unique(assign)
        # Sort by open cost descending (most expensive to close first)
        used_facilities = used_facilities[np.argsort(-open_cost[used_facilities])]
        
        for f_close in used_facilities:
            if time.time() > deadline - 2.0:
                break
            if load[f_close] == 0:
                continue
            
            customers = np.where(assign == f_close)[0]
            if len(customers) == 0:
                continue
                
            # Try to reassign these customers to other facilities
            # We need to check if it's possible and if it's worth it.
            # This is a small assignment problem.
            # For each customer in customers, find best alternative.
            # If all can be assigned, calculate total delta.
            
            temp_load = load.copy()
            temp_assign = assign.copy()
            temp_load[f_close] = 0
            
            success = True
            # Sort customers by demand descending to handle big ones first
            cust_order = customers[np.argsort(-demands[customers])]
            
            for c in cust_order:
                d_c = demands[c]
                # Find feasible facilities (excluding f_close)
                feas = (temp_load + d_c <= capacities)
                feas[f_close] = False
                if not np.any(feas):
                    success = False
                    break
                # Choose cheapest feasible
                costs = assign_cost[c, feas]
                best_f = np.argwhere(feas)[np.argmin(costs)][0]
                temp_assign[c] = best_f
                temp_load[best_f] += d_c
                
            if not success:
                continue
                
            # Calculate cost difference
            # Old cost part: open_cost[f_close] + sum(assign_cost[c, f_close] for c in customers)
            # New cost part: sum(assign_cost[c, temp_assign[c]] for c in customers) + (open costs of newly opened facilities)
            
            old_assign_sum = np.sum(assign_cost[customers, f_close])
            new_assign_sum = np.sum(assign_cost[customers, temp_assign[customers]])
            
            # Open cost change:
            # We save open_cost[f_close]
            # We might open new facilities.
            # Identify facilities that were empty before (in the context of this move) and are now used.
            # Actually, easier:
            old_used = np.unique(assign)
            new_used = np.unique(temp_assign)
            old_open_sum = np.sum(open_cost[old_used])
            new_open_sum = np.sum(open_cost[new_used])
            
            delta = (new_assign_sum - old_assign_sum) + (new_open_sum - old_open_sum)
            
            if delta < -1e-9:
                assign[:] = temp_assign
                load[:] = temp_load
                current_cost += delta
                improved = True
                break # Restart loop
        
    # Recalculate final cost to be precise
    current_cost = _calc_cost(demands, capacities, open_cost, assign_cost, assign)
    return assign, current_cost

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    n = len(demands)
    m = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_assign = None
    best_cost = np.inf
    
    # Multiple restarts with different initializations
    # 1. Greedy by demand (largest first)
    # 2. Greedy by random order
    # 3. Greedy by cost (cheapest to assign first)
    
    init_strategies = []
    
    # Strategy 1: Largest demand first
    order1 = np.argsort(-demands)
    init_strategies.append(order1)
    
    # Strategy 2: Random order
    rng = np.random.default_rng(42)
    order2 = rng.permutation(n)
    init_strategies.append(order2)
    
    # Strategy 3: Cheapest assignment cost first (sum of costs to all facilities)
    avg_costs = np.mean(assign_cost, axis=1)
    order3 = np.argsort(avg_costs)
    init_strategies.append(order3)
    
    for order in init_strategies:
        if time.time() > deadline - 10.0:
            break
            
        # Construct initial assignment
        curr_assign = np.full(n, -1, dtype=int)
        load = np.zeros(m, dtype=float)
        
        for i in order:
            d = demands[i]
            feas = load + d <= capacities
            if not np.any(feas):
                # Fallback: assign to least loaded
                f = np.argmin(load)
            else:
                costs = np.where(feas, assign_cost[i], np.inf)
                f = np.argmin(costs)
            curr_assign[i] = f
            load[f] += d
            
        if not np.all(curr_assign >= 0):
            continue
            
        # Local search
        curr_assign, curr_cost = _local_search(demands, capacities, open_cost, assign_cost, curr_assign, deadline - 5.0)
        
        if curr_cost < best_cost:
            best_cost = curr_cost
            best_assign = curr_assign.copy()
            
    # If we have time, do one more pass on the best solution with a longer local search
    if best_assign is not None and time.time() < deadline - 5.0:
        best_assign, best_cost = _local_search(demands, capacities, open_cost, assign_cost, best_assign, deadline - 1.0, max_iters=2000)
        
    if best_assign is None:
        # Fallback: assign all to facility 0 (might be invalid but better than crash)
        best_assign = np.zeros(n, dtype=int)
        
    return best_assign.astype(int)

# EVOLVE-BLOCK-END

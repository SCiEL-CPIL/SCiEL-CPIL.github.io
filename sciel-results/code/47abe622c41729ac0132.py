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

def _local_search(demands, capacities, open_cost, assign_cost, assign, deadline, max_iters=5000):
    """
    Vectorized local search with single moves, swap moves, and facility closing.
    Returns improved assign array and cost.
    """
    n = len(demands)
    m = len(capacities)
    assign = assign.copy().astype(int)
    load = np.zeros(m, dtype=float)
    for i in range(n):
        load[assign[i]] += demands[i]
    
    current_cost = _calc_cost(demands, capacities, open_cost, assign_cost, assign)
    
    # Precompute current assignment costs
    curr_assign_costs = assign_cost[np.arange(n), assign]
    
    improved = True
    iter_count = 0
    
    while improved and time.time() < deadline - 1.0 and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # --- Single Move Pass (Vectorized) ---
        # Calculate base deltas for assignment cost
        base_deltas = assign_cost - curr_assign_costs[:, np.newaxis]
        
        # Create mask for capacity feasibility
        feasible_cap = (load[np.newaxis, :] + demands[:, np.newaxis]) <= capacities[np.newaxis, :]
        
        # Mask out current facility
        curr_fac = assign[:, np.newaxis]
        is_current = (curr_fac == np.arange(m)[np.newaxis, :])
        feasible = feasible_cap & (~is_current)
        
        # Add open cost penalty for opening new facilities
        open_penalty = np.where(load[np.newaxis, :] == 0, open_cost[np.newaxis, :], 0.0)
        
        total_deltas = base_deltas + open_penalty
        total_deltas[~feasible] = np.inf
        
        # Find best move for each customer
        best_deltas = np.min(total_deltas, axis=1)
        best_facs = np.argmin(total_deltas, axis=1)
        
        # Find customers that can improve
        improve_mask = best_deltas < -1e-9
        if np.any(improve_mask):
            # Apply moves one by one to ensure capacity constraints are respected dynamically
            for i in np.where(improve_mask)[0]:
                f_new = best_facs[i]
                f_old = assign[i]
                
                # Re-check capacity
                if load[f_new] + demands[i] <= capacities[f_new]:
                    load[f_old] -= demands[i]
                    load[f_new] += demands[i]
                    assign[i] = f_new
                    curr_assign_costs[i] = assign_cost[i, f_new]
                    current_cost += best_deltas[i]
                    improved = True
                    
        if improved:
            continue  # Restart loop to recalculate deltas based on new loads
            
        # --- Swap Move Pass ---
        # Try more systematic swaps: all pairs of customers in different facilities
        # Sample a subset if too many pairs
        pairs_to_try = []
        # Group customers by facility
        fac_customers = {}
        for i in range(n):
            f = assign[i]
            if f not in fac_customers:
                fac_customers[f] = []
            fac_customers[f].append(i)
        
        # Try swaps between customers in different facilities
        # Limit to a reasonable number of random samples per iteration
        samples = min(2000, int((deadline - time.time()) * 500))  # ~2ms per sample
        rng = np.random.default_rng()
        
        for _ in range(samples):
            if time.time() > deadline - 1.0:
                break
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            f_i = assign[i]
            f_j = assign[j]
            if f_i == f_j:
                continue
            
            d_i = demands[i]
            d_j = demands[j]
            
            # Check capacity feasibility for swap
            if load[f_i] - d_i + d_j > capacities[f_i]:
                continue
            if load[f_j] - d_j + d_i > capacities[f_j]:
                continue
            
            # Calculate cost delta
            delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
            
            # Open cost delta
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
                break  # Restart loop for new state
        
        if improved:
            continue
            
        # --- Facility Closing Pass ---
        # Try to close a facility
        used_facilities = np.unique(assign)
        # Sort by open cost descending (most expensive to close first)
        used_facilities = used_facilities[np.argsort(-open_cost[used_facilities])]
        
        for f_close in used_facilities:
            if time.time() > deadline - 1.0:
                break
            if load[f_close] == 0:
                continue
            
            customers = np.where(assign == f_close)[0]
            if len(customers) == 0:
                continue
                
            # Try to reassign these customers to other facilities
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
            old_assign_sum = np.sum(assign_cost[customers, f_close])
            new_assign_sum = np.sum(assign_cost[customers, temp_assign[customers]])
            
            # Open cost change
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
                break  # Restart loop
        
    # Recalculate final cost to be precise
    current_cost = _calc_cost(demands, capacities, open_cost, assign_cost, assign)
    return assign, current_cost

def _polish_pass(demands, capacities, open_cost, assign_cost, assign, deadline):
    """
    Final high-precision polish pass with tighter tolerance and more iterations.
    Uses a smooth surrogate approach for the open cost term.
    """
    n = len(demands)
    m = len(capacities)
    assign = assign.copy().astype(int)
    load = np.zeros(m, dtype=float)
    for i in range(n):
        load[assign[i]] += demands[i]
    
    current_cost = _calc_cost(demands, capacities, open_cost, assign_cost, assign)
    curr_assign_costs = assign_cost[np.arange(n), assign]
    
    improved = True
    iter_count = 0
    max_iters = 10000
    
    while improved and time.time() < deadline - 0.5 and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # --- Single Move Pass with tighter tolerance ---
        base_deltas = assign_cost - curr_assign_costs[:, np.newaxis]
        feasible_cap = (load[np.newaxis, :] + demands[:, np.newaxis]) <= capacities[np.newaxis, :]
        curr_fac = assign[:, np.newaxis]
        is_current = (curr_fac == np.arange(m)[np.newaxis, :])
        feasible = feasible_cap & (~is_current)
        
        open_penalty = np.where(load[np.newaxis, :] == 0, open_cost[np.newaxis, :], 0.0)
        total_deltas = base_deltas + open_penalty
        total_deltas[~feasible] = np.inf
        
        best_deltas = np.min(total_deltas, axis=1)
        best_facs = np.argmin(total_deltas, axis=1)
        
        improve_mask = best_deltas < -1e-12  # Tighter tolerance
        if np.any(improve_mask):
            for i in np.where(improve_mask)[0]:
                f_new = best_facs[i]
                f_old = assign[i]
                if load[f_new] + demands[i] <= capacities[f_new]:
                    load[f_old] -= demands[i]
                    load[f_new] += demands[i]
                    assign[i] = f_new
                    curr_assign_costs[i] = assign_cost[i, f_new]
                    current_cost += best_deltas[i]
                    improved = True
        if improved:
            continue
        
        # --- Swap Move Pass with tighter tolerance ---
        rng = np.random.default_rng()
        samples = min(5000, int((deadline - time.time()) * 1000))
        
        for _ in range(samples):
            if time.time() > deadline - 0.5:
                break
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            f_i = assign[i]
            f_j = assign[j]
            if f_i == f_j:
                continue
            
            d_i = demands[i]
            d_j = demands[j]
            
            if load[f_i] - d_i + d_j > capacities[f_i]:
                continue
            if load[f_j] - d_j + d_i > capacities[f_j]:
                continue
            
            delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
            
            old_open_i = open_cost[f_i] if load[f_i] > 0 else 0.0
            old_open_j = open_cost[f_j] if load[f_j] > 0 else 0.0
            
            new_load_i = load[f_i] - d_i + d_j
            new_load_j = load[f_j] - d_j + d_i
            
            new_open_i = open_cost[f_i] if new_load_i > 0 else 0.0
            new_open_j = open_cost[f_j] if new_load_j > 0 else 0.0
            
            delta_open = (new_open_i + new_open_j) - (old_open_i + old_open_j)
            total_delta = delta_assign + delta_open
            
            if total_delta < -1e-12:
                assign[i], assign[j] = assign[j], assign[i]
                load[f_i] = new_load_i
                load[f_j] = new_load_j
                curr_assign_costs[i] = assign_cost[i, f_j]
                curr_assign_costs[j] = assign_cost[j, f_i]
                current_cost += total_delta
                improved = True
                break
        if improved:
            continue
        
        # --- Facility Closing Pass ---
        used_facilities = np.unique(assign)
        used_facilities = used_facilities[np.argsort(-open_cost[used_facilities])]
        
        for f_close in used_facilities:
            if time.time() > deadline - 0.5:
                break
            if load[f_close] == 0:
                continue
            
            customers = np.where(assign == f_close)[0]
            if len(customers) == 0:
                continue
                
            temp_load = load.copy()
            temp_assign = assign.copy()
            temp_load[f_close] = 0
            
            success = True
            cust_order = customers[np.argsort(-demands[customers])]
            
            for c in cust_order:
                d_c = demands[c]
                feas = (temp_load + d_c <= capacities)
                feas[f_close] = False
                if not np.any(feas):
                    success = False
                    break
                costs = assign_cost[c, feas]
                best_f = np.argwhere(feas)[np.argmin(costs)][0]
                temp_assign[c] = best_f
                temp_load[best_f] += d_c
                
            if not success:
                continue
                
            old_assign_sum = np.sum(assign_cost[customers, f_close])
            new_assign_sum = np.sum(assign_cost[customers, temp_assign[customers]])
            
            old_used = np.unique(assign)
            new_used = np.unique(temp_assign)
            old_open_sum = np.sum(open_cost[old_used])
            new_open_sum = np.sum(open_cost[new_used])
            
            delta = (new_assign_sum - old_assign_sum) + (new_open_sum - old_open_sum)
            
            if delta < -1e-12:
                assign[:] = temp_assign
                load[:] = temp_load
                current_cost += delta
                improved = True
                break
        
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
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_assign = None
    best_cost = np.inf
    
    # Multiple restarts with different initializations
    init_strategies = []
    
    # Strategy 1: Largest demand first
    order1 = np.argsort(-demands)
    init_strategies.append(order1)
    
    # Strategy 2: Random order
    rng = np.random.default_rng(42)
    order2 = rng.permutation(n)
    init_strategies.append(order2)
    
    # Strategy 3: Cheapest assignment cost first
    avg_costs = np.mean(assign_cost, axis=1)
    order3 = np.argsort(avg_costs)
    init_strategies.append(order3)
    
    # Strategy 4: Another random order
    rng2 = np.random.default_rng(123)
    order4 = rng2.permutation(n)
    init_strategies.append(order4)
    
    # Strategy 5: Sorted by variance of costs (less variable first)
    var_costs = np.var(assign_cost, axis=1)
    order5 = np.argsort(var_costs)
    init_strategies.append(order5)
    
    for order in init_strategies:
        if time.time() > deadline - 8.0:
            break
            
        # Construct initial assignment
        curr_assign = np.full(n, -1, dtype=int)
        load = np.zeros(m, dtype=float)
        
        for i in order:
            d = demands[i]
            feas = load + d <= capacities
            if not np.any(feas):
                f = np.argmin(load)
            else:
                costs = np.where(feas, assign_cost[i], np.inf)
                f = np.argmin(costs)
            curr_assign[i] = f
            load[f] += d
            
        if not np.all(curr_assign >= 0):
            continue
            
        # Local search
        curr_assign, curr_cost = _local_search(demands, capacities, open_cost, assign_cost, curr_assign, deadline - 3.0)
        
        if curr_cost < best_cost:
            best_cost = curr_cost
            best_assign = curr_assign.copy()
            
    # If we have time, do a polish pass on the best solution
    if best_assign is not None and time.time() < deadline - 2.0:
        best_assign, best_cost = _polish_pass(demands, capacities, open_cost, assign_cost, best_assign, deadline - 0.5)
        
    if best_assign is None:
        # Fallback: greedy assignment
        best_assign = np.zeros(n, dtype=int)
        load = np.zeros(m, dtype=float)
        for i in range(n):
            d = demands[i]
            feas = load + d <= capacities
            if np.any(feas):
                costs = np.where(feas, assign_cost[i], np.inf)
                f = np.argmin(costs)
            else:
                f = np.argmin(load)
            best_assign[i] = f
            load[f] += d
        
    return best_assign.astype(int)

# EVOLVE-BLOCK-END

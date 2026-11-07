"""Implemented with Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np


def _greedy_init(demands, capacities, assign_cost, rng):
    """
    Creates an initial feasible assignment using a greedy strategy.
    """
    n = len(demands)
    m = len(capacities)
    assignment = np.zeros(n, dtype=int)
    load = np.zeros(m)
    
    # Sort customers by demand descending to place large ones first
    order = np.argsort(-demands)
    
    for i in order:
        # Find feasible facilities
        feasible_mask = (load + demands[i]) <= (capacities + 1e-9)
        if not np.any(feasible_mask):
            # If no feasible facility found, return None
            return None, None
            
        # Pick the facility with minimum assignment cost among feasible
        costs = assign_cost[i].copy()
        costs[~feasible_mask] = np.inf
        f = np.argmin(costs)
        
        assignment[i] = f
        load[f] += demands[i]
        
    return assignment, load


def _calc_cost(assignment, demands, capacities, open_cost, assign_cost):
    """
    Calculate total cost: open costs + assignment costs.
    """
    n = len(assignment)
    m = len(capacities)
    
    # Check feasibility
    load = np.zeros(m)
    for i in range(n):
        load[assignment[i]] += demands[i]
    if np.any(load > capacities + 1e-9):
        return np.inf
        
    open_cost_sum = 0.0
    assign_cost_sum = 0.0
    
    used_facilities = set(assignment)
    for j in used_facilities:
        open_cost_sum += open_cost[j]
        
    for i in range(n):
        assign_cost_sum += assign_cost[i, assignment[i]]
        
    return open_cost_sum + assign_cost_sum


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n = len(demands)
    m = len(capacities)
    
    best_assignment = None
    best_cost = np.inf
    
    # Number of SA restarts
    n_restarts = 5
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        # 1. Generate initial solution
        init_res = _greedy_init(demands, capacities, assign_cost, rng)
        if init_res is None or init_res[0] is None:
            continue
            
        assignment, load = init_res
        current_cost = _calc_cost(assignment, demands, capacities, open_cost, assign_cost)
        
        if current_cost == np.inf:
            continue
            
        # 2. Simulated Annealing
        T_start = 1000.0
        T_end = 1.0
        alpha = 0.995  # Cooling rate
        max_iters = 20000
        it = 0
        
        while T_start > T_end and it < max_iters:
            it += 1
            T_start *= alpha
            
            # Generate neighbor: Move or Swap
            # 80% Move, 20% Swap
            if rng.random() < 0.8:
                # Move customer i to facility j
                i = rng.integers(n)
                cur_f = assignment[i]
                new_f = rng.integers(m)
                if new_f == cur_f:
                    continue
                    
                # Check capacity
                if load[new_f] + demands[i] > capacities[new_f] + 1e-9:
                    continue
                
                # Calculate delta
                delta_assign = assign_cost[i, new_f] - assign_cost[i, cur_f]
                delta_open = 0.0
                
                # Open cost changes
                # If cur_f becomes empty
                if load[cur_f] == demands[i]:
                    delta_open -= open_cost[cur_f]
                # If new_f was empty
                if load[new_f] == 0.0:
                    delta_open += open_cost[new_f]
                
                delta = delta_assign + delta_open
                
                # Acceptance
                if delta < 0 or rng.random() < np.exp(-delta / T_start):
                    # Perform move
                    load[cur_f] -= demands[i]
                    load[new_f] += demands[i]
                    assignment[i] = new_f
                    current_cost += delta
                    
            else:
                # Swap customers i and k
                i = rng.integers(n)
                k = rng.integers(n)
                if i == k:
                    continue
                    
                f_i = assignment[i]
                f_k = assignment[k]
                if f_i == f_k:
                    continue
                
                # Check capacities for swap
                # i moves from f_i to f_k, k moves from f_k to f_i
                # New load f_i: load[f_i] - d_i + d_k
                # New load f_k: load[f_k] - d_k + d_i
                
                if load[f_i] - demands[i] + demands[k] > capacities[f_i] + 1e-9:
                    continue
                if load[f_k] - demands[k] + demands[i] > capacities[f_k] + 1e-9:
                    continue
                
                # Calculate delta
                delta_assign = (assign_cost[i, f_k] - assign_cost[i, f_i]) + \
                               (assign_cost[k, f_i] - assign_cost[k, f_k])
                
                delta_open = 0.0
                
                # Open cost logic for swap
                old_open_i = open_cost[f_i] if load[f_i] > 0 else 0.0
                old_open_k = open_cost[f_k] if load[f_k] > 0 else 0.0
                
                new_load_i = load[f_i] - demands[i] + demands[k]
                new_load_k = load[f_k] - demands[k] + demands[i]
                
                new_open_i = open_cost[f_i] if new_load_i > 0 else 0.0
                new_open_k = open_cost[f_k] if new_load_k > 0 else 0.0
                
                delta_open = (new_open_i + new_open_k) - (old_open_i + old_open_k)
                
                delta = delta_assign + delta_open
                
                # Acceptance
                if delta < 0 or rng.random() < np.exp(-delta / T_start):
                    # Perform swap
                    load[f_i] = new_load_i
                    load[f_k] = new_load_k
                    assignment[i] = f_k
                    assignment[k] = f_i
                    current_cost += delta
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_assignment = assignment.copy()
            
    # Fallback if no valid solution found
    if best_assignment is None:
        init_res = _greedy_init(demands, capacities, assign_cost, np.random.default_rng(0))
        if init_res is not None and init_res[0] is not None:
            best_assignment = init_res[0]
        else:
            # Last resort: assign all to 0 (might be invalid, but better than crash)
            best_assignment = np.zeros(n, dtype=int)
            
    return best_assignment
# EVOLVE-BLOCK-END

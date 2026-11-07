"""Implemented with Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np


def _calculate_cost(assignment, open_cost, assign_cost):
    """Calculate total cost: open cost + assignment cost."""
    used_facilities = np.unique(assignment)
    open_cost_sum = np.sum(open_cost[used_facilities])
    assign_cost_sum = np.sum(assign_cost[np.arange(len(assignment)), assignment])
    return open_cost_sum + assign_cost_sum


def _simulate_annealing(initial_assignment, demands, capacities, open_cost, assign_cost, rng):
    """
    Runs Simulated Annealing starting from initial_assignment.
    """
    n = len(demands)
    m = len(capacities)
    
    # Current state
    current_assignment = initial_assignment.copy()
    current_load = np.zeros(m)
    for i in range(n):
        current_load[current_assignment[i]] += demands[i]
        
    current_cost = _calculate_cost(current_assignment, open_cost, assign_cost)
    
    # Best state
    best_assignment = current_assignment.copy()
    best_cost = current_cost
    
    # SA Parameters
    T_start = 500.0
    T_end = 1.0
    alpha = 0.995  # Cooling rate
    max_iters = 5000
    
    # Precompute indices for speed? No, just loop.
    
    for it in range(max_iters):
        T = T_start * (alpha ** it)
        if T < T_end:
            break
            
        # Pick a random customer
        i = rng.integers(n)
        cur_f = current_assignment[i]
        
        # Find feasible facilities for customer i
        # A facility j is feasible if load[j] + demands[i] <= capacities[j]
        # Note: cur_f is currently assigned, so we need to remove it first to check others,
        # or just check others directly.
        
        # To speed up, we can just check all other facilities.
        feasible_new_facs = []
        for j in range(m):
            if j == cur_f:
                continue
            if current_load[j] + demands[i] <= capacities[j] + 1e-9:
                feasible_new_facs.append(j)
                
        if not feasible_new_facs:
            continue
            
        # Pick a random new facility from feasible ones
        new_f = feasible_new_facs[rng.integers(len(feasible_new_facs))]
        
        # Calculate Delta Cost
        # Change in assignment cost
        d_assign = assign_cost[i, new_f] - assign_cost[i, cur_f]
        
        # Change in open cost
        d_open = 0.0
        # If cur_f becomes empty, we save open_cost[cur_f]
        if current_load[cur_f] == demands[i]:
            d_open -= open_cost[cur_f]
        # If new_f was empty, we pay open_cost[new_f]
        if current_load[new_f] == 0.0:
            d_open += open_cost[new_f]
            
        delta_cost = d_assign + d_open
        
        # Acceptance Probability
        if delta_cost < 0 or (rng.random() < np.exp(-delta_cost / T)):
            # Accept move
            current_assignment[i] = new_f
            current_load[cur_f] -= demands[i]
            current_load[new_f] += demands[i]
            current_cost += delta_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_assignment = current_assignment.copy()
                
    return best_assignment, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n = len(demands)
    m = len(capacities)
    
    best_assignment = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42) # Fixed seed for reproducibility within the run? Or random? 
    # Let's use random seed to explore different landscapes if called multiple times?
    # The evaluator calls assign() once per instance.
    # Using a fixed seed ensures deterministic behavior for the same input, which is good for debugging.
    # However, to avoid getting stuck in a specific local optimum for a specific instance structure,
    # randomization is good. But since we run SA, it's robust.
    # Let's use a random seed each time to ensure diversity if the evaluator re-runs?
    # Actually, standard practice is to be deterministic or use system time.
    # Let's use a random seed based on time or just random.
    # np.random.default_rng() uses OS entropy.
    
    # Initial Construction: Greedy
    # We'll try a few different greedy orders to start SA from different points.
    
    # Strategy 1: Sort by demand descending
    order1 = np.argsort(-demands)
    # Strategy 2: Random order
    order2 = rng.permutation(n)
    
    for order in [order1, order2]:
        init_assignment = np.zeros(n, dtype=int)
        load = np.zeros(m)
        feasible = True
        
        for i in order:
            # Find feasible facilities
            feasible_mask = (load + demands[i]) <= (capacities + 1e-9)
            if not np.any(feasible_mask):
                feasible = False
                break
            costs = assign_cost[i].copy()
            costs[~feasible_mask] = np.inf
            f = np.argmin(costs)
            init_assignment[i] = f
            load[f] += demands[i]
            
        if not feasible:
            continue
            
        # Run SA from this initial solution
        sa_assignment, sa_cost = _simulate_annealing(
            init_assignment, demands, capacities, open_cost, assign_cost, rng
        )
        
        if sa_cost < best_cost:
            best_cost = sa_cost
            best_assignment = sa_assignment.copy()
            
    return best_assignment
# EVOLVE-BLOCK-END

"""Program using Simulated Annealing for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Simulated Annealing.
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. Initial Solution: Greedy (Largest demand first, cheapest feasible)
    # This provides a valid starting point
    def greedy_init():
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        # Sort customers by demand descending
        order = np.argsort(-demands)
        for i in order:
            # Sort facilities by cost for this customer
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    res[i] = f
                    break
            else:
                # Fallback: assign to first facility (should not happen if feasible)
                res[i] = fac_order[0]
                load[fac_order[0]] += demands[i]
        return res

    current_assign = greedy_init()
    
    # Calculate initial cost
    def calc_cost(assignment):
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[assignment[i]] += demands[i]
        
        # Check capacity (should be valid, but just in case)
        if np.any(load > capacities + 1e-9):
            return np.inf
            
        cost = 0.0
        # Open costs
        for f in range(n_fac):
            if load[f] > 0:
                cost += open_cost[f]
        # Assignment costs
        cost += np.sum(assign_cost[np.arange(n_cust), assignment])
        return cost

    current_cost = calc_cost(current_assign)
    best_assign = current_assign.copy()
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    T = 500.0  # Initial temperature (scale relative to costs? costs are ~100s-1000s)
    # Let's estimate typical cost magnitude. 
    # assign_cost is 100 * dist. dist in [0, sqrt(2)]. So assign_cost in [0, 141].
    # 100 customers. Sum assign ~ 100 * 70 = 7000.
    # open_cost ~ 100 * 10 = 1000.
    # Total ~ 8000.
    # T should be high enough to accept bad moves initially.
    T = 2000.0
    T_min = 1e-4
    alpha = 0.995  # Cooling rate
    
    rng = np.random.default_rng(42)
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Generate a random neighbor
        # We can try:
        # 1. Move one customer to a random other facility
        # 2. Swap two customers assigned to different facilities
        
        move_type = rng.random()
        new_assign = current_assign.copy()
        
        if move_type < 0.7:
            # Single move
            i = rng.integers(0, n_cust)
            current_f = current_assign[i]
            # Pick a random other facility
            other_facs = [f for f in range(n_fac) if f != current_f]
            if other_facs:
                new_f = other_facs[rng.integers(0, len(other_facs))]
                new_assign[i] = new_f
        else:
            # Swap two customers
            i, j = rng.choice(n_cust, size=2, replace=False)
            if current_assign[i] != current_assign[j]:
                new_assign[i], new_assign[j] = current_assign[j], new_assign[i]
        
        # Check validity of new_assign
        # We need to check if capacity is violated.
        # Instead of recalculating full load, we can track load or just check the affected facilities.
        # For simplicity and safety, let's just calculate cost and check capacity.
        # Optimization: Pre-calculate load? 
        # Let's just do a quick capacity check for the modified facilities.
        
        # Actually, calculating full cost is O(N). N=100. 100 * 10000 steps = 1M ops. Fast.
        
        # Check capacity validity quickly
        # We can maintain a load array for current_assign to make this O(1) check for moves?
        # Let's try to be efficient.
        
        # Let's stick to simple O(N) check for now, but optimize if needed.
        # Given 120s limit, we can afford many iterations.
        
        new_load = np.zeros(n_fac)
        for k in range(n_cust):
            new_load[new_assign[k]] += demands[k]
            
        if np.any(new_load > capacities + 1e-9):
            # Invalid move, skip (or penalize heavily, but skipping is easier)
            T *= alpha
            continue
            
        new_cost = 0.0
        for f in range(n_fac):
            if new_load[f] > 0:
                new_cost += open_cost[f]
        new_cost += np.sum(assign_cost[np.arange(n_cust), new_assign])
        
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_assign = new_assign
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_assign = current_assign.copy()
                
        T *= alpha
        
    return best_assign

# EVOLVE-BLOCK-END

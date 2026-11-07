"""Implemented with Greedy Construction + Local Search (Move Heuristic)."""
# EVOLVE-BLOCK-START
import numpy as np


def _local_search(assignment, load, demands, capacities, open_cost, assign_cost, max_iters=1000):
    """
    Improves the assignment by moving customers to different facilities
    if it reduces the total cost (open + assignment).
    """
    n = len(demands)
    m = len(capacities)
    
    # Initial cost calculation
    def calc_cost(assignment, load):
        used = np.unique(assignment)
        return open_cost[used].sum() + assign_cost[np.arange(n), assignment].sum()
        
    current_cost = calc_cost(assignment, load)
    
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(n):
            cur_f = assignment[i]
            best_f = -1
            best_delta = 0.0  # We look for delta < 0
            
            cur_assign_cost = assign_cost[i, cur_f]
            
            for new_f in range(m):
                if new_f == cur_f:
                    continue
                # Check capacity
                if load[new_f] + demands[i] > capacities[new_f] + 1e-9:
                    continue
                
                # Calculate change in assignment cost
                d_assign = assign_cost[i, new_f] - cur_assign_cost
                
                # Calculate change in open cost
                d_open = 0.0
                # If current facility becomes empty, we save its open cost
                if load[cur_f] == demands[i]:
                    d_open -= open_cost[cur_f]
                # If new facility was empty, we pay its open cost
                if load[new_f] == 0.0:
                    d_open += open_cost[new_f]
                
                total_delta = d_assign + d_open
                
                if total_delta < best_delta - 1e-9:
                    best_delta = total_delta
                    best_f = new_f
                    
            if best_f != -1:
                # Perform the move
                load[cur_f] -= demands[i]
                load[best_f] += demands[i]
                assignment[i] = best_f
                current_cost += best_delta
                improved = True
                
    return assignment, current_cost


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n = len(demands)
    m = len(capacities)
    
    best_assignment = None
    best_cost = np.inf
    
    # Define multiple construction strategies to avoid local optima
    # 1. Demand Descending
    # 2. Demand Ascending
    # 3. Random Order
    rng = np.random.default_rng()
    strategies = [
        np.argsort(-demands),
        np.argsort(demands),
        rng.permutation(n)
    ]
    
    for order in strategies:
        assignment = np.zeros(n, dtype=int)
        load = np.zeros(m)
        
        # Greedy Construction
        for i in order:
            # Find feasible facilities
            feasible_mask = (load + demands[i]) <= (capacities + 1e-9)
            if not np.any(feasible_mask):
                # Should not happen for valid instances, but handle gracefully
                # Assign to facility with most remaining capacity?
                # For now, skip or break. 
                # Let's try to find any facility with space, even if not strictly feasible in this step?
                # No, must be feasible.
                continue
            
            # Among feasible, pick min assign_cost
            costs = assign_cost[i].copy()
            costs[~feasible_mask] = np.inf
            f = np.argmin(costs)
            
            assignment[i] = f
            load[f] += demands[i]
        
        # Local Search
        res_assignment, res_cost = _local_search(assignment, load, demands, capacities, open_cost, assign_cost)
        
        if res_cost < best_cost:
            best_cost = res_cost
            best_assignment = res_assignment.copy()
            
    return best_assignment
# EVOLVE-BLOCK-END

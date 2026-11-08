"""Capacitated facility location: greedy seed + vectorized local search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using greedy initialization followed by
    vectorized local search with relocate moves.
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    def compute_score(res):
        """Compute total cost of an assignment."""
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf
        used = load > 0
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
    
    def greedy_init():
        """Greedy initialization: assign customers in order of demand (largest first)."""
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    break
            else:
                # Fallback: assign to facility with most remaining space
                space = capacities - load
                best_f = np.argmax(space)
                if space[best_f] >= demands[i]:
                    res[i] = best_f
                    load[best_f] += demands[i]
                else:
                    res[i] = fac_order[0]
        return res
    
    def local_search(res, load, max_passes=500):
        """
        Local search with relocate moves.
        Tries to improve by moving one customer at a time.
        Uses vectorized cost computation for efficiency.
        """
        improved = True
        it = 0
        while improved and it < max_passes:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            
            # Try relocate moves for each customer in random order
            for i in np.random.permutation(n_cust):
                if time.time() > deadline:
                    return
                
                current_f = res[i]
                d_i = demands[i]
                
                # Find best alternative facility
                best_f = -1
                best_delta = 1e-9
                
                # Vectorized computation of deltas for all facilities
                # delta = assign_cost[i, f] - assign_cost[i, current_f] + open_cost changes
                base_delta = assign_cost[i, :] - assign_cost[i, current_f]
                
                # Open cost changes:
                # If f is currently unused (load[f] == 0), we add open_cost[f]
                # If current_f becomes empty (load[current_f] == d_i), we subtract open_cost[current_f]
                
                # Create mask of feasible facilities (enough capacity)
                feasible = (load + d_i <= capacities + 1e-9)
                feasible[current_f] = False  # Skip current facility
                
                if not np.any(feasible):
                    continue
                
                # Compute delta for feasible facilities
                deltas = base_delta.copy()
                
                # Add open cost for facilities that would be newly opened
                newly_opened = feasible & (load == 0)
                deltas[newly_opened] += open_cost[newly_opened]
                
                # Subtract open cost if current facility becomes empty
                if load[current_f] == d_i:
                    deltas[feasible] -= open_cost[current_f]
                
                # Find best feasible facility
                feasible_indices = np.where(feasible)[0]
                if len(feasible_indices) == 0:
                    continue
                
                min_idx = np.argmin(deltas[feasible_indices])
                best_f = feasible_indices[min_idx]
                best_delta = deltas[best_f]
                
                if best_delta < -1e-9:  # Strict improvement
                    # Apply move
                    load[current_f] -= d_i
                    load[best_f] += d_i
                    res[i] = best_f
                    improved = True
    
    # Initial greedy solution
    res = greedy_init()
    load = np.zeros(n_fac)
    np.add.at(load, res, demands)
    
    # Run local search on initial solution
    local_search(res, load)
    best_score = compute_score(res)
    best_assign = res.copy()
    
    # Try a few more local search passes to see if we can improve further
    for attempt in range(3):
        if time.time() > deadline:
            break
        
        # Perturb the solution slightly to escape local optima
        res_perturbed = best_assign.copy()
        load_perturbed = np.zeros(n_fac)
        np.add.at(load_perturbed, res_perturbed, demands)
        
        # Move 5 random customers to their best feasible alternative
        for _ in range(5):
            if time.time() > deadline:
                break
            i = np.random.randint(n_cust)
            current_f = res_perturbed[i]
            d_i = demands[i]
            
            feasible = (load_perturbed + d_i <= capacities + 1e-9)
            feasible[current_f] = False
            
            if np.any(feasible):
                feasible_indices = np.where(feasible)[0]
                # Choose randomly among feasible to add diversity
                new_f = feasible_indices[np.random.randint(len(feasible_indices))]
                load_perturbed[current_f] -= d_i
                load_perturbed[new_f] += d_i
                res_perturbed[i] = new_f
        
        # Run local search on perturbed solution
        local_search(res_perturbed, load_perturbed, max_passes=200)
        score = compute_score(res_perturbed)
        
        if score < best_score:
            best_score = score
            best_assign = res_perturbed.copy()
    
    return best_assign.astype(int)
# EVOLVE-BLOCK-END

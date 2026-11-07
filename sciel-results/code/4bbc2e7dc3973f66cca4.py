"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    timeout = 140.0  # Safety margin below 150s
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays for ease of use
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    def time_exceeded():
        return time.time() - start_time > timeout
    
    def calc_total_cost(assignment):
        """Calculate total cost: assignment costs + open costs for used facilities."""
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
        used = np.unique(assignment)
        open_cost_sum = np.sum(open_cost[used])
        return assign_cost_sum + open_cost_sum
    
    def is_valid(assignment):
        """Check if assignment is valid (all customers assigned, capacity constraints met)."""
        if len(assignment) != n_customers:
            return False
        if np.any(assignment < 0) or np.any(assignment >= n_facilities):
            return False
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        return np.all(load <= capacities + 1e-6)
    
    def greedy_init(strategy='largest'):
        """Greedy initialization. Strategy: 'largest' (largest demand first) or 'nearest'."""
        res = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        if strategy == 'largest':
            order = np.argsort(-demands)
        else:
            # Sort by minimum possible assignment cost
            min_costs = np.min(assign_cost, axis=1)
            order = np.argsort(min_costs)
        
        for i in order:
            d = demands[i]
            remaining_cap = capacities - load
            feasible_mask = remaining_cap >= d
            if not np.any(feasible_mask):
                # No feasible facility, pick one with most remaining capacity
                f = int(np.argmax(remaining_cap))
            else:
                # Consider open cost: if facility is new, add open cost
                costs = np.where(feasible_mask, assign_cost[i], np.inf)
                # Add open cost for facilities not yet used
                used_mask = load > 0
                costs[~used_mask] += open_cost[~used_mask]
                f = int(np.argmin(costs))
            res[i] = f
            load[f] += d
        return res
    
    def local_search(assignment, max_iterations=100000):
        """Local search with 1-moves and 2-swaps."""
        assignment = assignment.copy()
        current_cost = calc_total_cost(assignment)
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        
        improved = True
        iterations = 0
        while improved and not time_exceeded() and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # 1-moves: try moving each customer to a different facility
            for i in range(n_customers):
                if time_exceeded():
                    break
                current_f = assignment[i]
                d_i = demands[i]
                
                for new_f in range(n_facilities):
                    if new_f == current_f:
                        continue
                    
                    # Check capacity: can new facility accommodate?
                    if load[new_f] + d_i > capacities[new_f] + 1e-6:
                        continue
                    
                    # Calculate cost delta
                    # Remove from current_f: save assign_cost[i, current_f]
                    # Add to new_f: pay assign_cost[i, new_f]
                    delta = assign_cost[i, new_f] - assign_cost[i, current_f]
                    
                    # Open cost changes
                    # If current_f becomes empty, save open_cost[current_f]
                    # If new_f was empty, pay open_cost[new_f]
                    old_load_current = load[current_f] - d_i
                    old_load_new = load[new_f]
                    
                    if old_load_current <= 0 and current_f != new_f:
                        delta -= open_cost[current_f]  # Save open cost
                    if old_load_new <= 0:
                        delta += open_cost[new_f]  # Pay open cost
                    
                    if delta < -1e-9:
                        # Accept move
                        load[current_f] -= d_i
                        load[new_f] += d_i
                        assignment[i] = new_f
                        current_cost += delta
                        improved = True
                        break  # Restart scan after a move
            
            # 2-swaps: try swapping two customers between facilities
            if not time_exceeded():
                for i in range(n_customers):
                    if time_exceeded():
                        break
                    for j in range(i + 1, n_customers):
                        if time_exceeded():
                            break
                        f_i = assignment[i]
                        f_j = assignment[j]
                        if f_i == f_j:
                            continue
                        
                        d_i = demands[i]
                        d_j = demands[j]
                        
                        # Check capacity after swap
                        # f_i loses d_i, gains d_j
                        # f_j loses d_j, gains d_i
                        new_load_fi = load[f_i] - d_i + d_j
                        new_load_fj = load[f_j] - d_j + d_i
                        
                        if new_load_fi > capacities[f_i] + 1e-6:
                            continue
                        if new_load_fj > capacities[f_j] + 1e-6:
                            continue
                        
                        # Calculate cost delta
                        # i moves from f_i to f_j
                        # j moves from f_j to f_i
                        delta = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                (assign_cost[j, f_i] - assign_cost[j, f_j])
                        
                        # Open cost changes
                        # f_i was used, still used (gains j)
                        # f_j was used, still used (gains i)
                        # No open cost change since both facilities remain used
                        
                        if delta < -1e-9:
                            # Accept swap
                            load[f_i] = new_load_fi
                            load[f_j] = new_load_fj
                            assignment[i] = f_j
                            assignment[j] = f_i
                            current_cost += delta
                            improved = True
                            break  # Restart after a swap
                    if improved:
                        break
        
        return assignment, current_cost
    
    # Try multiple initialization strategies and pick the best
    best_assignment = None
    best_cost = np.inf
    
    # Strategy 1: Largest demand first
    if not time_exceeded():
        init1 = greedy_init('largest')
        if is_valid(init1):
            ls1, cost1 = local_search(init1)
            if is_valid(ls1):
                if cost1 < best_cost:
                    best_cost = cost1
                    best_assignment = ls1.copy()
    
    # Strategy 2: Nearest first
    if not time_exceeded():
        init2 = greedy_init('nearest')
        if is_valid(init2):
            ls2, cost2 = local_search(init2)
            if is_valid(ls2):
                if cost2 < best_cost:
                    best_cost = cost2
                    best_assignment = ls2.copy()
    
    # If we still have time, do more local search passes with random perturbations
    if best_assignment is not None and not time_exceeded():
        for restart in range(20):
            if time_exceeded():
                break
            
            # Perturb the best solution: randomly reassign 5-10 customers
            perturbed = best_assignment.copy()
            n_perturb = random.randint(5, 10)
            indices = random.sample(range(n_customers), n_perturb)
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[perturbed[i]] += demands[i]
            
            for idx in indices:
                d = demands[idx]
                remaining_cap = capacities - load
                feasible_mask = remaining_cap >= d
                if np.any(feasible_mask):
                    f = int(np.random.choice(np.where(feasible_mask)[0]))
                else:
                    f = int(np.argmax(remaining_cap))
                load[perturbed[idx]] -= d
                perturbed[idx] = f
                load[f] += d
            
            if is_valid(perturbed):
                ls_res, ls_cost = local_search(perturbed)
                if is_valid(ls_res):
                    if ls_cost < best_cost:
                        best_cost = ls_cost
                        best_assignment = ls_res.copy()
    
    if best_assignment is None:
        # Fallback: simple greedy
        best_assignment = greedy_init('largest')
    
    return best_assignment.astype(int)
# EVOLVE-BLOCK-END

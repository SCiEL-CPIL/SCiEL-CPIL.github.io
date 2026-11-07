"""Capacitated facility location solver with local search and LNS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _greedy_assign(demands, capacities, open_cost, assign_cost):
    """Greedy: largest demand first, cheapest feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in np.argsort(-demands):
        # Sort facilities by cost for this customer
        order = np.argsort(assign_cost[i])
        for f in order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # If no facility has capacity, assign to the one with most remaining capacity
            remaining = capacities - load
            f = np.argmax(remaining)
            load[f] += demands[i]
            result[i] = f
    return result


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Check validity
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            return float('inf')
        load[f] += demands[i]
    
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            return float('inf')
    
    # Compute cost
    open_cost_total = 0.0
    assign_cost_total = 0.0
    for f in range(n_facilities):
        if load[f] > 0:
            open_cost_total += open_cost[f]
    
    for i in range(n_customers):
        assign_cost_total += assign_cost[i, result[i]]
    
    return open_cost_total + assign_cost_total


def _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=5000):
    """Local search: try moving single customers to other facilities."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    if not np.isfinite(best_cost):
        return best_result
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[best_result[i]] += demands[i]
    
    improved = True
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        
        # Try moving each customer to a better facility
        for i in range(n_customers):
            current_f = best_result[i]
            current_load_f = load[current_f]
            
            # Find best alternative facility
            # Consider facilities with lower marginal cost
            # Marginal cost of moving i to f:
            # - Remove i from current_f: save assign_cost[i, current_f]
            # - If current_f becomes empty, save open_cost[current_f]
            # - Add i to f: pay assign_cost[i, f]
            # - If f was empty, pay open_cost[f]
            
            best_alt_f = -1
            best_delta = 0
            
            for f in range(n_facilities):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f] + 1e-9:
                    continue
                
                # Compute delta cost
                delta = -assign_cost[i, current_f] + assign_cost[i, f]
                
                # Open cost changes
                if current_load_f == demands[i]:
                    # current_f becomes empty
                    delta -= open_cost[current_f]
                if load[f] == 0:
                    # f becomes non-empty
                    delta += open_cost[f]
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_alt_f = f
            
            if best_alt_f >= 0:
                # Perform the move
                load[current_f] -= demands[i]
                load[best_alt_f] += demands[i]
                best_result[i] = best_alt_f
                best_cost += best_delta
                improved = True
    
    return best_result


def _pair_swap_search(result, demands, capacities, open_cost, assign_cost, max_iter=2000):
    """Try swapping pairs of customers between facilities."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    if not np.isfinite(best_cost):
        return best_result
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[best_result[i]] += demands[i]
    
    improved = True
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        
        # Try swapping pairs of customers
        for i in range(n_customers):
            fi = best_result[i]
            for j in range(i + 1, n_customers):
                fj = best_result[j]
                if fi == fj:
                    continue
                
                # Check if swap is feasible
                if load[fi] - demands[i] + demands[j] > capacities[fi] + 1e-9:
                    continue
                if load[fj] - demands[j] + demands[i] > capacities[fj] + 1e-9:
                    continue
                
                # Compute delta cost
                delta = -assign_cost[i, fi] - assign_cost[j, fj] + assign_cost[i, fj] + assign_cost[j, fi]
                
                # Open cost changes
                # fi after swap: load[fi] - demands[i] + demands[j]
                # fj after swap: load[fj] - demands[j] + demands[i]
                new_load_fi = load[fi] - demands[i] + demands[j]
                new_load_fj = load[fj] - demands[j] + demands[i]
                
                # Current open cost contribution
                current_open = (1 if load[fi] > 0 else 0) * open_cost[fi] + (1 if load[fj] > 0 else 0) * open_cost[fj]
                # New open cost contribution
                new_open = (1 if new_load_fi > 0 else 0) * open_cost[fi] + (1 if new_load_fj > 0 else 0) * open_cost[fj]
                
                delta += new_open - current_open
                
                if delta < -1e-9:
                    # Perform swap
                    load[fi] -= demands[i]
                    load[fj] -= demands[j]
                    load[fi] += demands[j]
                    load[fj] += demands[i]
                    best_result[i] = fj
                    best_result[j] = fi
                    best_cost += delta
                    improved = True
    
    return best_result


def _lncs(result, demands, capacities, open_cost, assign_cost, num_restarts=10):
    """Large Neighborhood Search with randomized restarts."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    if not np.isfinite(best_cost):
        return best_result
    
    rng = np.random.RandomState(42)
    
    for restart in range(num_restarts):
        # Perturb the solution
        perturbed = best_result.copy()
        # Randomly reassign a subset of customers
        subset_size = max(1, n_customers // 10)
        indices = rng.choice(n_customers, size=subset_size, replace=False)
        
        # For each customer in the subset, reassign greedily
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            if i not in indices:
                load[perturbed[i]] += demands[i]
        
        for i in indices:
            order = np.argsort(assign_cost[i])
            assigned = False
            for f in order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    perturbed[i] = f
                    assigned = True
                    break
            if not assigned:
                remaining = capacities - load
                f = np.argmax(remaining)
                load[f] += demands[i]
                perturbed[i] = f
        
        # Local search on perturbed solution
        improved = _local_search(perturbed, demands, capacities, open_cost, assign_cost, max_iter=2000)
        cost = _compute_cost(improved, demands, capacities, open_cost, assign_cost)
        
        if np.isfinite(cost) and cost < best_cost - 1e-9:
            best_result = improved
            best_cost = cost
    
    return best_result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities minimizing total cost."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Phase 1: Greedy initial solution
    result = _greedy_assign(demands, capacities, open_cost, assign_cost)
    
    # Phase 2: Local search
    if time.time() < deadline - 30:
        result = _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=10000)
    
    # Phase 3: Pair swap search
    if time.time() < deadline - 20:
        result = _pair_swap_search(result, demands, capacities, open_cost, assign_cost, max_iter=5000)
    
    # Phase 4: More local search
    if time.time() < deadline - 10:
        result = _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=5000)
    
    # Phase 5: LNS if time remains
    if time.time() < deadline - 5:
        result = _lncs(result, demands, capacities, open_cost, assign_cost, num_restarts=5)
    
    # Final local search pass
    if time.time() < deadline - 1:
        result = _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=2000)
    
    # Ensure validity
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            # Fallback to greedy
            result = _greedy_assign(demands, capacities, open_cost, assign_cost)
            break
        load[f] += demands[i]
    
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            # Fallback to greedy
            result = _greedy_assign(demands, capacities, open_cost, assign_cost)
            break
    
    return result
# EVOLVE-BLOCK-END

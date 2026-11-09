"""Capacitated facility location: Lagrangian relaxation + enhanced local search with time budget."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _true_obj(result, demands, capacities, open_cost, assign_cost):
    """Compute objective: open costs + assignment costs. Returns inf if infeasible."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        return float('inf')
    open_sum = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            open_sum += open_cost[f]
    assign_sum = float(np.sum(assign_cost[np.arange(len(result)), result]))
    return open_sum + assign_sum


def _greedy_seed(demands, capacities, assign_cost):
    """Largest-first greedy assignment (fallback)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        feasible = []
        for f in np.argsort(assign_cost[i]):
            if load[f] + d <= capacities[f]:
                feasible.append(f)
        if feasible:
            result[i] = feasible[0]
            load[result[i]] += d
        else:
            result[i] = 0
            load[0] += d
    return result


def _lagrangian_seed(demands, capacities, assign_cost, open_cost, max_iters=300):
    """
    Lagrangian relaxation approach:
    - Dualize capacity constraints with multipliers lambda
    - For given lambda, solve relaxed problem (each customer picks min cost facility)
    - Update lambda via subgradient method
    - Return best feasible solution found
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Initialize multipliers
    lam = np.zeros(n_fac)
    
    best_result = None
    best_obj = float('inf')
    
    # Step size parameter
    alpha = 10.0
    
    for it in range(max_iters):
        # Solve relaxed problem: for each customer, pick facility minimizing
        # assign_cost[c, f] + lam[f] * demands[c]
        # If no customer assigned to f, don't pay open_cost (handled in dual update)
        
        # Cost matrix for relaxed problem
        # effective_cost[c, f] = assign_cost[c, f] + lam[f] * demands[c]
        effective_cost = assign_cost + lam[np.newaxis, :] * demands[:, np.newaxis]
        
        # Each customer picks their best facility
        result = np.argmin(effective_cost, axis=1)
        
        # Compute load
        load = np.zeros(n_fac)
        for c in range(n_cust):
            load[result[c]] += demands[c]
        
        # Check feasibility
        if np.all(load <= capacities):
            obj = _true_obj(result, demands, capacities, open_cost, assign_cost)
            if obj < best_obj:
                best_obj = obj
                best_result = result.copy()
            
            # If feasible, we can still try to improve with different lambda
            # But for now, continue to see if we find better
        
        # Subgradient update: subgradient for lambda[f] is load[f] - capacities[f]
        subgrad = load - capacities
        
        # Update lambda (project onto non-negative orthant)
        lam = np.maximum(0, lam + alpha * subgrad)
        
        # Decay step size
        alpha *= 0.97
    
    # If no feasible solution found, fall back to greedy
    if best_result is None:
        return _greedy_seed(demands, capacities, assign_cost)
    
    return best_result


def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """Enhanced local search: single moves, pair swaps, and 3-customer cycles."""
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    
    best_obj = _true_obj(result, demands, capacities, open_cost, assign_cost)
    if best_obj == float('inf'):
        return result, best_obj
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # Try single moves: move customer c from result[c] to facility f
        for c in range(n_cust):
            if time.time() >= deadline:
                break
            d = demands[c]
            cur_f = result[c]
            cur_cost_c = assign_cost[c, cur_f]
            
            # Try all other facilities
            for f in range(n_fac):
                if f == cur_f:
                    continue
                # Check capacity: new facility has room
                if load[f] + d > capacities[f]:
                    continue
                
                # Compute delta in objective
                # Assignment cost change
                delta_assign = assign_cost[c, f] - cur_cost_c
                
                # Open cost change:
                # If cur_f loses its last customer, we save open_cost[cur_f]
                # If f gains its first customer, we pay open_cost[f]
                delta_open = 0.0
                if load[cur_f] == d:  # cur_f becomes empty
                    delta_open -= open_cost[cur_f]
                if load[f] == 0:  # f becomes non-empty
                    delta_open += open_cost[f]
                
                delta = delta_assign + delta_open
                
                if delta < -1e-9:
                    # Apply move
                    result[c] = f
                    load[cur_f] -= d
                    load[f] += d
                    best_obj += delta
                    improved = True
                    no_improve = 0
                    break  # Move this customer to next customer
        
        if not improved:
            no_improve += 1
        
        # Try pair swaps: swap customers c1 and c2
        if no_improve < max_no_improve // 2 and time.time() < deadline:
            for c1 in range(n_cust):
                if time.time() >= deadline:
                    break
                for c2 in range(c1 + 1, n_cust):
                    if time.time() >= deadline:
                        break
                    f1 = result[c1]
                    f2 = result[c2]
                    if f1 == f2:
                        continue
                    d1 = demands[c1]
                    d2 = demands[c2]
                    
                    # After swap: c1 goes to f2, c2 goes to f1
                    # Check capacities
                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1
                    if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                        continue
                    
                    # Compute delta
                    delta_assign = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                                   (assign_cost[c1, f1] + assign_cost[c2, f2])
                    
                    # Open cost changes
                    delta_open = 0.0
                    if load[f1] > 0 and new_load_f1 == 0:
                        delta_open -= open_cost[f1]
                    elif load[f1] == 0 and new_load_f1 > 0:
                        delta_open += open_cost[f1]
                    if load[f2] > 0 and new_load_f2 == 0:
                        delta_open -= open_cost[f2]
                    elif load[f2] == 0 and new_load_f2 > 0:
                        delta_open += open_cost[f2]
                    
                    delta = delta_assign + delta_open
                    
                    if delta < -1e-9:
                        # Apply swap
                        result[c1], result[c2] = f2, f1
                        load[f1] = new_load_f1
                        load[f2] = new_load_f2
                        best_obj += delta
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # Try 3-customer rotations: c1->f2, c2->f3, c3->f1
        if no_improve < max_no_improve // 3 and time.time() < deadline:
            # Sample a subset of triples to avoid O(n^3) explosion
            for c1 in range(n_cust):
                if time.time() >= deadline:
                    break
                for c2 in range(c1 + 1, n_cust):
                    if time.time() >= deadline:
                        break
                    for c3 in range(c2 + 1, min(c2 + 20, n_cust)):  # Limit inner loop
                        if time.time() >= deadline:
                            break
                        f1 = result[c1]
                        f2 = result[c2]
                        f3 = result[c3]
                        
                        # All three must be on different facilities
                        if f1 == f2 or f2 == f3 or f1 == f3:
                            continue
                        
                        d1 = demands[c1]
                        d2 = demands[c2]
                        d3 = demands[c3]
                        
                        # After rotation: c1->f2, c2->f3, c3->f1
                        # Check capacities
                        new_load_f1 = load[f1] - d1 + d3
                        new_load_f2 = load[f2] - d2 + d1
                        new_load_f3 = load[f3] - d3 + d2
                        
                        if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2] or new_load_f3 > capacities[f3]:
                            continue
                        
                        # Compute delta
                        delta_assign = (assign_cost[c1, f2] + assign_cost[c2, f3] + assign_cost[c3, f1]) - \
                                       (assign_cost[c1, f1] + assign_cost[c2, f2] + assign_cost[c3, f3])
                        
                        # Open cost changes
                        delta_open = 0.0
                        for f_idx, new_l in [(f1, new_load_f1), (f2, new_load_f2), (f3, new_load_f3)]:
                            old_l = load[f_idx]
                            if old_l > 0 and new_l == 0:
                                delta_open -= open_cost[f_idx]
                            elif old_l == 0 and new_l > 0:
                                delta_open += open_cost[f_idx]
                        
                        delta = delta_assign + delta_open
                        
                        if delta < -1e-9:
                            # Apply rotation
                            result[c1], result[c2], result[c3] = f2, f3, f1
                            load[f1] = new_load_f1
                            load[f2] = new_load_f2
                            load[f3] = new_load_f3
                            best_obj += delta
                            improved = True
                            no_improve = 0
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            no_improve += 1
    
    return result, best_obj


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian relaxation seed + enhanced local search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for 145s max
    deadline = time.time() + 145.0
    
    # Lagrangian relaxation seed
    result = _lagrangian_seed(demands, capacities, assign_cost, open_cost, max_iters=300)
    
    # Enhanced local search with time budget
    result, best_obj = _local_search(result, demands, capacities, open_cost, assign_cost, deadline)
    
    # Verify feasibility
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        # Fallback: return greedy result
        return _greedy_seed(demands, capacities, assign_cost)
    
    return result.astype(int)
# EVOLVE-BLOCK-END

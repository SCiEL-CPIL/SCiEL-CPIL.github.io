"""Capacitated facility location: Lagrangian + Vectorized Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _total_cost(choice, demands, capacities, open_cost, assign_cost):
    """Compute the total cost of an assignment."""
    n, m = assign_cost.shape
    used = np.zeros(m, dtype=bool)
    used[choice] = True
    open_sum = np.sum(open_cost[used])
    assign_sum = np.sum(assign_cost[np.arange(n), choice])
    return open_sum + assign_sum

def _is_valid(choice, demands, capacities):
    """Check if the assignment respects capacity constraints."""
    load = np.zeros(len(capacities))
    np.add.at(load, choice, demands)
    return np.all(load <= capacities + 1e-9)

def _greedy_init(demands, capacities, open_cost, assign_cost, rng):
    """Greedy initialization: assign largest demands first to cheapest feasible facility."""
    n, m = assign_cost.shape
    choice = np.zeros(n, dtype=int)
    load = np.zeros(m)
    # Sort customers by demand descending
    order = np.argsort(-demands)
    for c in order:
        d = demands[c]
        # Find feasible facilities
        feasible = np.where(capacities - load >= d)[0]
        if len(feasible) == 0:
            # Fallback: assign to facility with most free space (might violate capacity if infeasible)
            # But problem states capacities are sufficient for a feasible solution to exist usually.
            # If not, we pick the one with max remaining capacity.
            f = int(np.argmax(capacities - load))
        else:
            # Among feasible, pick the one with lowest assignment cost
            costs = assign_cost[c, feasible]
            f = feasible[np.argmin(costs)]
        choice[c] = f
        load[f] += d
    return choice

def _random_feasible_init(demands, capacities, open_cost, assign_cost, rng):
    """Random feasible initialization with cost-based weighting."""
    n, m = assign_cost.shape
    choice = np.zeros(n, dtype=int)
    load = np.zeros(m)
    order = np.argsort(-demands)
    for c in order:
        d = demands[c]
        feasible = np.where(capacities - load >= d)[0]
        if len(feasible) == 0:
            f = int(np.argmax(capacities - load))
        else:
            # Weight by inverse cost
            weights = 1.0 / (assign_cost[c, feasible] + 1e-6)
            weights /= np.sum(weights)
            f = feasible[rng.choice(len(feasible), p=weights)]
        choice[c] = f
        load[f] += d
    return choice

def _lagrangian_init(demands, capacities, open_cost, assign_cost, deadline, rng):
    """Lagrangian relaxation to find a good initial solution."""
    n, m = assign_cost.shape
    lam = np.zeros(m)
    best_sol = None
    best_cost = np.inf
    step = 1.0
    
    for k in range(500):
        if time.time() > deadline - 5.0:
            break
        # Subproblem: min sum_c min_f (assign_cost[c,f] + lam[f]) + sum_f max(0, load_f - cap_f) * penalty?
        # Standard Lagrangian for CFLP relaxes capacity constraints.
        # Dual function g(lam) = sum_c min_f (assign_cost[c,f] + lam[f]) + sum_f lam[f]*(cap_f) ... wait.
        # Let's use a simpler heuristic approach based on Lagrangian multipliers for capacity.
        # Adjusted cost for customer c to facility f: assign_cost[c,f] + lam[f]
        adjusted = assign_cost + lam
        choice = np.argmin(adjusted, axis=1)
        
        load = np.zeros(m)
        np.add.at(load, choice, demands)
        
        # Check feasibility
        if np.all(load <= capacities + 1e-9):
            cost = _total_cost(choice, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_sol = choice.copy()
        
        # Update multipliers
        # grad = load - capacities
        grad = load - capacities
        # Step size
        if best_cost > np.sum(np.min(adjusted, axis=1)):
            gap = best_cost - np.sum(np.min(adjusted, axis=1))
            step = 2.0 * gap / (np.dot(grad, grad) + 1e-12)
        
        lam = np.maximum(lam + step * grad, 0.0)
        
        # Occasionally try to repair and keep best
        if k % 20 == 0 and best_sol is not None:
             # We just track the best feasible found in the loop
             pass
             
    return best_sol

def _repair(choice, demands, capacities, assign_cost):
    """Repair infeasible assignment by moving customers from overloaded facilities."""
    n, m = assign_cost.shape
    choice = choice.copy()
    load = np.zeros(m)
    np.add.at(load, choice, demands)
    
    # While any facility is over capacity
    guard = 0
    while np.any(load > capacities + 1e-9) and guard < 1000:
        guard += 1
        # Find overloaded facilities
        over_facilities = np.where(load > capacities + 1e-9)[0]
        if len(over_facilities) == 0:
            break
        
        # Pick the most overloaded facility
        f_bad = over_facilities[np.argmax(load[over_facilities])]
        
        # Find customers in this facility
        custs = np.where(choice == f_bad)[0]
        if len(custs) == 0:
             break
             
        # Move the customer with the largest demand to the cheapest feasible facility
        # Or move the one that minimizes cost increase? Let's try largest demand first to free up space.
        c = custs[np.argmax(demands[custs])]
        d = demands[c]
        
        # Find feasible facilities (excluding f_bad)
        free_cap = capacities - load
        free_cap[f_bad] = -np.inf
        feasible = np.where(free_cap >= d)[0]
        
        if len(feasible) == 0:
            # If no feasible facility, we are in trouble. 
            # Try to move to facility with most free space even if it violates? 
            # No, we must maintain validity. 
            # If we can't repair, return invalid.
            # But problem guarantees feasible instance. 
            # Maybe we need to move multiple customers? 
            # For now, break.
            break
        
        # Pick cheapest feasible
        costs = assign_cost[c, feasible]
        f_new = feasible[np.argmin(costs)]
        
        # Move
        choice[c] = f_new
        load[f_bad] -= d
        load[f_new] += d
        
    return choice

def _local_search(choice, demands, capacities, open_cost, assign_cost, deadline):
    """Vectorized local search with single moves and 2-swaps."""
    n, m = assign_cost.shape
    choice = choice.copy()
    load = np.zeros(m)
    np.add.at(load, choice, demands)
    
    current_cost = _total_cost(choice, demands, capacities, open_cost, assign_cost)
    best_cost = current_cost
    best_choice = choice.copy()
    
    # Precompute indices for vectorization
    idx = np.arange(n)
    
    while time.time() < deadline - 2.0:
        improved = False
        
        # 1. Single Moves
        # For each customer, find the best facility to move to.
        # We can vectorize the cost delta calculation.
        # delta_cost(c, f) = assign_cost[c, f] - assign_cost[c, choice[c]] 
        #                    + open_cost_delta(c, f)
        
        # To handle open cost delta efficiently:
        # If we move c from f0 to f1:
        # - If load[f0] becomes 0, we save open_cost[f0].
        # - If load[f1] was 0, we pay open_cost[f1].
        
        # This is hard to fully vectorize across all customers at once because 
        # "load[f0] becomes 0" depends on the specific move.
        # However, we can iterate customers and vectorize over facilities.
        
        for c in range(n):
            if time.time() > deadline - 2.0:
                break
                
            f0 = choice[c]
            d = demands[c]
            
            # Feasible facilities: load[f] + d <= capacities[f]
            # And f != f0
            feasible_mask = (load + d) <= capacities
            feasible_mask[f0] = False
            
            if not np.any(feasible_mask):
                continue
                
            feasible_idx = np.where(feasible_mask)[0]
            
            # Calculate assignment cost delta
            assign_delta = assign_cost[c, feasible_idx] - assign_cost[c, f0]
            
            # Calculate open cost delta
            # Source f0: if load[f0] - d == 0, we save open_cost[f0]
            save_open = 0.0
            if load[f0] - d <= 1e-9:
                save_open = -open_cost[f0]
                
            # Target f: if load[f] == 0, we pay open_cost[f]
            pay_open = np.where(load[feasible_idx] <= 1e-9, open_cost[feasible_idx], 0.0)
            
            total_delta = assign_delta + pay_open + save_open
            
            # Find best move
            min_idx = np.argmin(total_delta)
            min_delta = total_delta[min_idx]
            
            if min_delta < -1e-9:
                f_new = feasible_idx[min_idx]
                
                # Apply move
                load[f0] -= d
                load[f_new] += d
                choice[c] = f_new
                
                current_cost += min_delta
                improved = True
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_choice = choice.copy()
                    
        # 2. Two-Swaps
        # Try swapping two customers from different facilities.
        # This can be expensive, so we sample a limited number of pairs.
        num_swaps_to_try = 200
        for _ in range(num_swaps_to_try):
            if time.time() > deadline - 2.0:
                break
                
            # Pick two random customers
            c1, c2 = np.random.choice(n, 2, replace=False)
            f1 = choice[c1]
            f2 = choice[c2]
            
            if f1 == f2:
                continue
                
            d1 = demands[c1]
            d2 = demands[c2]
            
            # Check capacity after swap
            # f1 loses d1, gains d2. New load: load[f1] - d1 + d2
            # f2 loses d2, gains d1. New load: load[f2] - d2 + d1
            if load[f1] - d1 + d2 > capacities[f1] + 1e-9:
                continue
            if load[f2] - d2 + d1 > capacities[f2] + 1e-9:
                continue
                
            # Calculate cost delta
            # Assign cost delta:
            # c1 moves f1 -> f2: assign_cost[c1, f2] - assign_cost[c1, f1]
            # c2 moves f2 -> f1: assign_cost[c2, f1] - assign_cost[c2, f2]
            assign_delta = (assign_cost[c1, f2] - assign_cost[c1, f1]) + \
                           (assign_cost[c2, f1] - assign_cost[c2, f2])
            
            # Open cost delta
            # Check if f1 becomes empty
            new_load_f1 = load[f1] - d1 + d2
            new_load_f2 = load[f2] - d2 + d1
            
            old_load_f1 = load[f1]
            old_load_f2 = load[f2]
            
            open_delta = 0.0
            # f1
            if old_load_f1 > 1e-9 and new_load_f1 <= 1e-9:
                open_delta -= open_cost[f1] # Saved
            elif old_load_f1 <= 1e-9 and new_load_f1 > 1e-9:
                open_delta += open_cost[f1] # Paid
            # f2
            if old_load_f2 > 1e-9 and new_load_f2 <= 1e-9:
                open_delta -= open_cost[f2]
            elif old_load_f2 <= 1e-9 and new_load_f2 > 1e-9:
                open_delta += open_cost[f2]
                
            total_delta = assign_delta + open_delta
            
            if total_delta < -1e-9:
                # Apply swap
                load[f1] = new_load_f1
                load[f2] = new_load_f2
                choice[c1] = f2
                choice[c2] = f1
                
                current_cost += total_delta
                improved = True
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_choice = choice.copy()
                    
        if not improved:
            break
            
    return best_choice, best_cost

def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n, m = assign_cost.shape
    rng = np.random.default_rng(42)
    
    best_solution = None
    best_cost = np.inf
    
    # Generate initial candidates
    candidates = []
    
    # 1. Greedy
    try:
        cand_greedy = _greedy_init(demands, capacities, open_cost, assign_cost, rng)
        if _is_valid(cand_greedy, demands, capacities):
            candidates.append(cand_greedy)
        else:
            # Repair
            cand_repaired = _repair(cand_greedy, demands, capacities, assign_cost)
            if _is_valid(cand_repaired, demands, capacities):
                candidates.append(cand_repaired)
    except Exception:
        pass
        
    # 2. Lagrangian
    try:
        cand_lag = _lagrangian_init(demands, capacities, open_cost, assign_cost, deadline, rng)
        if cand_lag is not None and _is_valid(cand_lag, demands, capacities):
            candidates.append(cand_lag)
    except Exception:
        pass
        
    # 3. Random Feasible
    try:
        cand_rand = _random_feasible_init(demands, capacities, open_cost, assign_cost, rng)
        if _is_valid(cand_rand, demands, capacities):
            candidates.append(cand_rand)
    except Exception:
        pass
        
    # If no valid candidates, force a greedy one
    if not candidates:
        cand_greedy = _greedy_init(demands, capacities, open_cost, assign_cost, rng)
        cand_repaired = _repair(cand_greedy, demands, capacities, assign_cost)
        candidates.append(cand_repaired)
        
    # Run local search on each candidate
    for cand in candidates:
        if time.time() > deadline - 5.0:
            break
            
        try:
            sol, cost = _local_search(cand, demands, capacities, open_cost, assign_cost, deadline)
            if _is_valid(sol, demands, capacities):
                if cost < best_cost:
                    best_cost = cost
                    best_solution = sol
        except Exception:
            continue
            
    if best_solution is None:
        # Fallback to first candidate
        best_solution = candidates[0]
        
    return np.asarray(best_solution, dtype=int)

# EVOLVE-BLOCK-END

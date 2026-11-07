"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost for a given assignment."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        if 0 <= f < n_fac:
            load[f] += demands[i]
        else:
            return np.inf
    
    if np.any(load > capacities + 1e-6):
        return np.inf
    
    # Opening costs
    open_facs = np.unique(result)
    open_cost_sum = np.sum(open_cost[open_facs])
    
    # Assignment costs
    assign_cost_sum = np.sum(assign_cost[np.arange(len(result)), result])
    
    return open_cost_sum + assign_cost_sum

def _is_valid(result, demands, capacities):
    """Check if assignment is valid (capacity constraints satisfied)."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return False
        load[f] += demands[i]
    return np.all(load <= capacities + 1e-6)

def _greedy_init(demands, capacities, open_cost, assign_cost, mode='cost'):
    """
    Greedy initialization.
    mode: 'cost' - assign to cheapest feasible facility
          'demand' - assign largest demands first to cheapest feasible
          'random' - random valid assignment
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    if mode == 'random':
        # Simple random assignment, repair if invalid
        # This is a weak init, but provides diversity
        for i in range(n_cust):
            valid_facs = [f for f in range(n_fac) if load[f] + demands[i] <= capacities[f]]
            if valid_facs:
                f = random.choice(valid_facs)
                result[i] = f
                load[f] += demands[i]
            else:
                # Fallback to facility with most space
                rem = capacities - load
                f = np.argmax(rem)
                result[i] = f
                load[f] += demands[i]
        return result

    # Sort customers
    if mode == 'demand':
        order = np.argsort(-demands)
    else:
        order = np.argsort(assign_cost.mean(axis=1)) # Sort by average cost? Or just index order?
        # Let's stick to index order for 'cost' mode to be deterministic, 
        # or sort by min cost.
        min_costs = assign_cost.min(axis=1)
        order = np.argsort(min_costs)
        
    for i in order:
        best_f = -1
        best_cost = np.inf
        
        # Find feasible facilities
        feasible = (load + demands[i] <= capacities)
        if not np.any(feasible):
            # No feasible facility, assign to one with most remaining capacity (might violate, handled later)
            rem = capacities - load
            best_f = np.argmax(rem)
            result[i] = best_f
            load[best_f] += demands[i]
            continue
            
        # Among feasible, pick min cost
        # We add a small penalty for opening a new facility if load is 0
        # To encourage consolidation? No, greedy should just pick min assign cost.
        # Opening cost is fixed per facility.
        
        costs = assign_cost[i, feasible]
        # If load is 0, we might want to consider open cost.
        # But for simplicity, let's just minimize assign cost first.
        # Actually, minimizing assign cost is usually good.
        
        min_idx = np.argmin(costs)
        # Map back to original index
        feasible_indices = np.where(feasible)[0]
        best_f = feasible_indices[min_idx]
        
        result[i] = best_f
        load[best_f] += demands[i]
        
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit=10.0):
    """
    Local search with 1-moves and 2-swaps.
    """
    n_cust = len(result)
    n_fac = len(capacities)
    
    if not _is_valid(result, demands, capacities):
        # Try to repair
        # Simple repair: move customers from overloaded facilities
        load = np.zeros(n_fac)
        for i, f in enumerate(result):
            load[f] += demands[i]
        
        for f in range(n_fac):
            if load[f] > capacities[f]:
                # Find customers in f, move them to other facilities with space
                custs_in_f = [i for i in range(n_cust) if result[i] == f]
                # Sort by demand descending to move big ones first? Or smallest?
                # Moving smallest is easier to fit.
                custs_in_f.sort(key=lambda x: demands[x])
                for i in custs_in_f:
                    # Find a facility with space
                    moved = False
                    for new_f in range(n_fac):
                        if new_f == f:
                            continue
                        if load[new_f] + demands[i] <= capacities[new_f]:
                            # Move
                            result[i] = new_f
                            load[f] -= demands[i]
                            load[new_f] += demands[i]
                            moved = True
                            break
                    if not moved:
                        # Still overloaded, assign to facility with most space
                        rem = capacities - load
                        new_f = np.argmax(rem)
                        result[i] = new_f
                        load[f] -= demands[i]
                        load[new_f] += demands[i]
                        
        if not _is_valid(result, demands, capacities):
            return result # Return invalid, caller will handle

    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
        
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result

    start_time = time.time()
    improved = True
    
    # Precompute facility customer lists for swap search
    # This is O(N) to build, but updated after moves.
    # For efficiency, we'll rebuild or update carefully.
    
    while improved:
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        
        # 1. Try 1-moves
        # For each customer, try moving to every other facility
        for i in range(n_cust):
            old_f = result[i]
            old_load = load[old_f]
            
            # If moving this customer closes the facility, we save open_cost
            # We need to know if it's the last customer in old_f
            # Efficient check: if load[old_f] == demands[i], it might be the last one.
            # But we need to be sure no other customer is there.
            # Since we are iterating, we can just check the count.
            # Let's maintain a count array.
            
            # Try all other facilities
            best_delta = 0
            best_new_f = -1
            
            for new_f in range(n_fac):
                if new_f == old_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                
                # Delta open
                delta_open = 0
                # Old facility
                if old_load == demands[i]:
                    # Check if any other customer is in old_f
                    # This is O(N) if done naively.
                    # Let's use a helper or just accept the cost for now.
                    # Optimization: maintain fac_counts
                    pass # We'll use a simpler heuristic: just compute cost delta if it looks promising?
                    # Or just compute full cost delta for the best few candidates?
                    # Let's stick to O(N) check for now, N=100 is small.
                    if not np.any(result == old_f) or np.sum(result == old_f) == 1:
                         # If it was the only one, we close it
                         # Wait, if np.sum(result == old_f) == 1, and we are moving i, then it becomes 0.
                         # So we save open_cost[old_f]
                         delta_open -= open_cost[old_f]
                         
                # New facility
                if load[new_f] == 0:
                    # We open it
                    delta_open += open_cost[new_f]
                    
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_new_f = new_f
            
            if best_new_f != -1 and best_delta < -1e-9:
                # Accept move
                result[i] = best_new_f
                load[old_f] -= demands[i]
                load[best_new_f] += demands[i]
                current_cost += best_delta
                improved = True
                # Restart scan? Or continue? 
                # Continuing is faster, but might miss improvements.
                # Let's continue.
        
        if time.time() - start_time > time_limit:
            break
            
        # 2. Try 2-swaps
        # Group customers by facility
        fac_customers = {f: [] for f in range(n_fac)}
        for i in range(n_cust):
            fac_customers[result[i]].append(i)
            
        for f1 in range(n_fac):
            if len(fac_customers[f1]) < 1:
                continue
            for f2 in range(f1 + 1, n_fac):
                if len(fac_customers[f2]) < 1:
                    continue
                
                # Try swapping i in f1 with j in f2
                for i in fac_customers[f1]:
                    for j in fac_customers[f2]:
                        # Check capacity
                        if load[f1] - demands[i] + demands[j] > capacities[f1]:
                            continue
                        if load[f2] - demands[j] + demands[i] > capacities[f2]:
                            continue
                        
                        delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                        
                        delta_open = 0
                        # Check if f1 becomes empty
                        new_load_f1 = load[f1] - demands[i] + demands[j]
                        new_load_f2 = load[f2] - demands[j] + demands[i]
                        
                        # If f1 had only i, and we move i out, but j comes in, it doesn't become empty unless j is also moved out? No, j comes IN.
                        # So f1 load changes from L to L - d_i + d_j.
                        # It becomes empty only if L - d_i + d_j == 0.
                        # Since d_j > 0, L - d_i must be < 0, which is impossible if L >= d_i.
                        # Wait, if L == d_i (only i in f1), then new load is d_j. Not empty.
                        # So f1 doesn't close.
                        # Similarly f2 doesn't close.
                        # UNLESS: we are swapping such that one facility becomes empty?
                        # If f1 had only i, and we move i to f2, and j (from f2) to f1.
                        # f1 load: 0 + d_j = d_j. Not empty.
                        # f2 load: L2 - d_j + d_i.
                        # So 2-swaps NEVER change the set of open facilities?
                        # Correct! Swapping customers between two non-empty facilities keeps both non-empty.
                        # Swapping a customer from a facility with 1 customer to another, and a customer from the other to the first, keeps both non-empty.
                        # The only way to change open status is a 1-move (moving the last customer out).
                        # So delta_open is 0 for 2-swaps.
                        
                        if delta_assign < -1e-9:
                            # Accept swap
                            result[i] = f2
                            result[j] = f1
                            load[f1] -= demands[i]
                            load[f1] += demands[j]
                            load[f2] -= demands[j]
                            load[f2] += demands[i]
                            current_cost += delta_assign
                            improved = True
                            # Update fac_customers? 
                            # It's complex to update efficiently. 
                            # Let's break and rebuild next iteration.
                            # Or just break out of loops.
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return result

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s buffer
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_result = None
    best_cost = np.inf
    
    # 1. Generate multiple initial solutions
    inits = []
    
    # Greedy by cost
    r1 = _greedy_init(demands, capacities, open_cost, assign_cost, mode='cost')
    inits.append(r1)
    
    # Greedy by demand
    r2 = _greedy_init(demands, capacities, open_cost, assign_cost, mode='demand')
    inits.append(r2)
    
    # Random init (a few times)
    for _ in range(5):
        r_rand = _greedy_init(demands, capacities, open_cost, assign_cost, mode='random')
        inits.append(r_rand)
        
    # 2. Local search on each
    # Allocate time per init
    # Total time ~140s.
    # 7 inits. ~20s each.
    
    for r in inits:
        if time.time() > deadline:
            break
            
        # Quick validity check and repair
        if not _is_valid(r, demands, capacities):
             # Skip or try to repair? 
             # _local_search handles repair at start.
             pass
             
        # Run local search
        # Give it a time slice
        time_slice = 15.0
        start_ls = time.time()
        refined = _local_search(r, demands, capacities, open_cost, assign_cost, time_limit=time_slice)
        
        if _is_valid(refined, demands, capacities):
            cost = _compute_cost(refined, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = refined.copy()
                
    # 3. If we have time left, do more local search on the best result
    if best_result is not None and time.time() < deadline - 10.0:
        remaining_time = deadline - 10.0 - time.time()
        if remaining_time > 5.0:
            # Polish the best result
            polished = _local_search(best_result, demands, capacities, open_cost, assign_cost, time_limit=remaining_time)
            if _is_valid(polished, demands, capacities):
                cost = _compute_cost(polished, demands, capacities, open_cost, assign_cost)
                if cost < best_cost:
                    best_result = polished
                    best_cost = cost
                    
    # 4. Final fallback
    if best_result is None or not _is_valid(best_result, demands, capacities):
        # Emergency greedy
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost, mode='demand')
        # Try to repair
        if not _is_valid(best_result, demands, capacities):
             # Hard repair
             load = np.zeros(n_fac)
             for i, f in enumerate(best_result):
                 load[f] += demands[i]
             for f in range(n_fac):
                 if load[f] > capacities[f]:
                     custs = [i for i in range(n_cust) if best_result[i] == f]
                     custs.sort(key=lambda x: demands[x])
                     for i in custs:
                         # Move to facility with most space
                         rem = capacities - load
                         new_f = np.argmax(rem)
                         best_result[i] = new_f
                         load[f] -= demands[i]
                         load[new_f] += demands[i]

    return best_result

# EVOLVE-BLOCK-END

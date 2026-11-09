"""Capacitated facility location: greedy seed + vectorized local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Uses greedy seed followed by vectorized local search with time budget.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Time budget: leave safety margin (150s limit, use 130s)
    start_time = time.time()
    time_budget = 130.0
    
    # Convert to numpy arrays
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    def true_obj(assignment):
        """Compute the true objective: open costs + assignment costs."""
        # Open costs for facilities with at least one customer
        used = np.zeros(n_facilities, dtype=bool)
        for f in assignment:
            used[f] = True
        open_sum = np.sum(open_cost[used])
        # Assignment costs
        assign_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
        return open_sum + assign_sum
    
    def greedy_seed():
        """Greedy: largest demand first, cheapest feasible facility."""
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities, dtype=np.float64)
        
        # Sort customers by demand descending
        order = np.argsort(-demands)
        for i in order:
            # Sort facilities by assignment cost for this customer
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
            else:
                # Should not happen if feasible, but handle gracefully
                # Assign to facility with most remaining capacity
                rem = capacities - load
                f = np.argmax(rem)
                load[f] += demands[i]
                result[i] = f
        return result
    
    # Initial solution
    best_assign = greedy_seed()
    best_obj = true_obj(best_assign)
    
    # Local search: try moving each customer to a different facility
    # Use vectorized evaluation for speed
    def local_search(assignment, obj, max_time=None):
        """Improve assignment via single-customer moves."""
        assignment = assignment.copy()
        obj = obj
        load = np.zeros(n_facilities, dtype=np.float64)
        for f in assignment:
            load[f] += demands[f]  # Wait, this is wrong - load is per facility
        # Recompute load properly
        load = np.zeros(n_facilities, dtype=np.float64)
        for c in range(n_customers):
            load[assignment[c]] += demands[c]
        
        # Track which facilities are "open" (have at least one customer)
        open_count = np.zeros(n_facilities, dtype=np.int64)
        for f in assignment:
            open_count[f] += 1
        
        improved = True
        while improved:
            improved = False
            if max_time is not None:
                if time.time() - start_time > max_time:
                    break
            
            # For each customer, find the best alternative facility
            # We'll do this in batches for vectorization
            for c in range(n_current_customers if False else n_customers):
                current_f = assignment[c]
                current_cost = assign_cost[c, current_f]
                
                # Compute delta for moving customer c from current_f to each other facility
                # New cost = assign_cost[c, f] - current_cost
                # Capacity check: load[f] + demands[c] <= capacities[f]
                # Open cost change: if moving from current_f and current_f becomes empty, save open_cost[current_f]
                # If moving to f and f was empty, add open_cost[f]
                
                # Vectorized: compute all possible moves
                # delta_assign = assign_cost[c, :] - current_cost
                # But we need to account for open cost changes
                
                # For efficiency, precompute for this customer
                deltas = assign_cost[c, :].copy()
                # Subtract current assignment cost
                deltas[current_f] = 0  # No change if staying
                
                # Open cost changes
                # If current_f has only this customer, moving away saves open_cost[current_f]
                # If target f has no customers, moving there adds open_cost[f]
                
                # This is tricky to vectorize fully because open cost depends on counts
                # Let's do a semi-vectorized approach
                
                # For each facility f != current_f:
                #   delta = assign_cost[c, f] - current_cost
                #   if open_count[current_f] == 1: delta -= open_cost[current_f]
                #   if open_count[f] == 0: delta += open_cost[f]
                #   Check capacity: load[f] + demands[c] <= capacities[f]
                
                best_delta = 0  # 0 means no improvement (stay)
                best_f = current_f
                
                # Iterate over facilities (only 20, so this is fast)
                for f in range(n_facilities):
                    if f == current_f:
                        continue
                    if load[f] + demands[c] > capacities[f]:
                        continue
                    
                    delta = assign_cost[c, f] - current_cost
                    if open_count[current_f] == 1:
                        delta -= open_cost[current_f]
                    if open_count[f] == 0:
                        delta += open_cost[f]
                    
                    if delta < best_delta - 1e-10:
                        best_delta = delta
                        best_f = f
                
                if best_f != current_f:
                    # Make the move
                    old_f = current_f
                    new_f = best_f
                    
                    # Update load
                    load[old_f] -= demands[c]
                    load[new_f] += demands[c]
                    
                    # Update open counts
                    open_count[old_f] -= 1
                    open_count[new_f] += 1
                    
                    # Update assignment
                    assignment[c] = new_f
                    
                    # Update objective
                    obj += best_delta
                    
                    improved = True
        
        return assignment, obj
    
    # Run local search with time budget
    max_time_for_ls = 120.0  # Leave 10s safety margin
    best_assign, best_obj = local_search(best_assign, best_obj, max_time_for_ls)
    
    # If we have time, try a few more restarts with different greedy strategies
    remaining_time = 130.0 - (time.time() - start_time)
    if remaining_time > 20.0:
        # Try a different greedy: randomize tie-breaking
        np.random.seed(42)
        for restart in range(3):
            if time.time() - start_time > 120.0:
                break
            
            # Different greedy: smallest cost first, but with some randomization
            result = np.full(n_customers, -1, dtype=int)
            load = np.zeros(n_facilities, dtype=np.float64)
            
            # Random order with slight preference for large demands
            order = np.argsort(-demands + np.random.randn(n_customers) * 5)
            for i in order:
                fac_order = np.argsort(assign_cost[i] + np.random.randn(n_facilities) * 1.0)
                for f in fac_order:
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        break
                else:
                    rem = capacities - load
                    f = np.argmax(rem)
                    load[f] += demands[i]
                    result[i] = f
            
            # Local search on this seed
            candidate, cand_obj = local_search(result, true_obj(result), 120.0 - (time.time() - start_time))
            
            if cand_obj < best_obj:
                best_assign = candidate
                best_obj = cand_obj
    
    return best_assign.astype(int)
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Initialization: Greedy (Largest Demand First)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    open_status = np.zeros(n_facilities, dtype=bool)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        d = demands[i]
        # Find cheapest feasible facility
        costs = assign_cost[i]
        # We need to check capacity
        feasible = load + d <= capacities
        if not np.any(feasible):
            # Should not happen if problem is feasible, but handle gracefully
            # Assign to facility with most remaining capacity
            f = np.argmax(capacities - load)
        else:
            # Mask infeasible ones with inf
            masked_costs = np.where(feasible, costs, np.inf)
            f = np.argmin(masked_costs)
            
        result[i] = f
        load[f] += d
        open_status[f] = True

    # 2. Local Search: Move and Swap
    # We track the total cost to quickly evaluate improvements
    current_cost = np.sum(assign_cost[np.arange(n_customers), result]) + np.sum(open_cost[open_status])
    
    deadline = time.time() + 110.0 # Safety margin
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # --- Move Step ---
        # Iterate customers in random order to avoid bias? Or fixed?
        # Fixed order is fine for small N.
        for i in range(n_customers):
            f_old = result[i]
            d = demands[i]
            
            # Try all other facilities
            for f_new in range(n_facilities):
                if f_new == f_old:
                    continue
                
                # Check capacity
                if load[f_new] + d > capacities[f_new]:
                    continue
                
                # Calculate Delta Cost
                delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                delta_open = 0.0
                
                # Check if f_old becomes empty
                new_load_old = load[f_old] - d
                if new_load_old == 0:
                    delta_open -= open_cost[f_old]
                
                # Check if f_new was empty
                if load[f_new] == 0:
                    delta_open += open_cost[f_new]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9: # Improvement
                    # Perform move
                    load[f_old] -= d
                    load[f_new] += d
                    result[i] = f_new
                    
                    if load[f_old] == 0:
                        open_status[f_old] = False
                    if load[f_new] > 0: # Always true after add
                        open_status[f_new] = True
                    
                    current_cost += total_delta
                    improved = True
                    # Restart loop? Or continue? 
                    # Continuing is fine, but restarting ensures we don't miss
                    # interactions. However, for speed, we can just continue
                    # to next customer. Best Improvement is better but slower.
                    # First Improvement is faster. Let's stick to first improvement 
                    # per customer pass, then repeat.
                    break # Move to next customer
        
        # --- Swap Step ---
        # Only if no moves were made in this pass? Or always?
        # Swaps are more expensive (O(N^2 * F)). With N=100, N^2=10000.
        # 10000 * 20 checks = 200,000 operations per pass. Very fast.
        # We will do swaps if no move was found in the previous move-pass 
        # OR we can just alternate.
        # Let's do swaps if no move was made.
        if not improved:
            # Check for swaps
            # To speed up, we can restrict swaps to customers in the same or nearby facilities?
            # No, global swaps are needed.
            # We can limit swaps to a random subset if N is large, but N=100 is small.
            
            # Optimization: Only consider swaps if it might help.
            # A swap is beneficial if assign_cost[i, f_j] + assign_cost[j, f_i] < assign_cost[i, f_i] + assign_cost[j, f_j]
            # And capacity allows.
            
            # We can pre-compute potential gains?
            # For now, brute force is fast enough.
            
            for i in range(n_customers):
                f_i = result[i]
                d_i = demands[i]
                
                for j in range(i + 1, n_customers):
                    f_j = result[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    # Check capacity for swap
                    # i moves to f_j, j moves to f_i
                    # New load f_i: load[f_i] - d_i + d_j
                    # New load f_j: load[f_j] - d_j + d_i
                    
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    # Calculate Delta Cost
                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                                   (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    # Opening costs don't change in a swap (set of open facilities remains same)
                    # UNLESS one facility becomes empty?
                    # If load[f_i] - d_i + d_j == 0, then f_i closes.
                    # If load[f_j] - d_j + d_i == 0, then f_j closes.
                    # But since d_i, d_j > 0, if they swap, the loads change.
                    # If f_i had only i, and we move i to f_j, f_i load becomes 0 (if d_j=0? No d_j>0).
                    # Wait. If f_i had only i, load[f_i]=d_i.
                    # After swap: load[f_i] = d_i - d_i + d_j = d_j.
                    # So f_i is still open (unless d_j=0, which is not possible).
                    # So opening status never changes in a swap.
                    
                    if delta_assign < -1e-9:
                        # Perform swap
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        result[i] = f_j
                        result[j] = f_i
                        current_cost += delta_assign
                        improved = True
                        break # Restart swap loop? Or break inner?
                if improved:
                    break # Break outer loop to restart pass
        
    return result
# EVOLVE-BLOCK-END

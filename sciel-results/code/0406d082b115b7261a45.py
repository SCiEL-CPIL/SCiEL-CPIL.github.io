"""Capacitated facility location: Vectorized multi-start local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a local search strategy.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for 140s max
    deadline = time.time() + 140.0
    
    # Precompute customer order by demand (largest first) for initialization stability
    # But the core algorithm is vectorized, so we'll use random restarts with different 
    # initializations.
    
    best_score = np.inf
    best_result = np.zeros(n_cust, dtype=int)
    
    # Initialize a valid solution using a greedy heuristic (fast)
    # This ensures we always have a valid fallback.
    result_greedy = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    # Sort customers by demand descending for better greedy packing
    order = np.argsort(-demands)
    for i in order:
        # Find feasible facilities
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            # This shouldn't happen if total capacity >= total demand, 
            # but if it does, we have to pick the one with most remaining capacity
            # or just fail. The problem implies a valid solution exists.
            # For safety, pick the facility with max remaining capacity.
            remaining = capacities - load
            f = np.argmax(remaining)
            if load[f] + demands[i] > capacities[f]:
                # Invalid instance? Return something, though it will be invalid.
                # But the contract says valid instances.
                pass
            result_greedy[i] = f
            load[f] += demands[i]
        else:
            # Among feasible, pick cheapest assignment cost
            costs = assign_cost[i, feasible]
            f_idx = feasible[np.argmin(costs)]
            result_greedy[i] = f_idx
            load[f_idx] += demands[i]
            
    best_result = result_greedy.copy()
    # Calculate score for greedy
    used_fac = np.unique(best_result)
    score_greedy = np.sum(open_cost[used_fac]) + np.sum(assign_cost[np.arange(n_cust), best_result])
    best_score = score_greedy

    # Local Search Strategy:
    # 1. Assign each customer to the facility that minimizes (assign_cost + penalty).
    # 2. Penalty encourages balancing load or avoiding opening expensive facilities.
    # A common heuristic is to treat open_cost as a fixed charge.
    # Since we can't easily do fixed-charge LPs, we use a "potential" method.
    
    # We will try multiple "temperature" or "bias" parameters for the open cost.
    # Effective cost for customer i to facility f:
    # c[i, f] = assign_cost[i, f] + lambda * open_cost[f] / expected_load[f] ?
    # Or simpler: iterate assignment, then check capacity. If capacity violated, 
    # move the customer with highest marginal cost increase to another facility.
    
    # Let's use a simpler "Alternating Minimization" with capacity repair.
    # 1. Compute effective cost matrix.
    # 2. Assign all to argmin.
    # 3. Check capacities.
    # 4. If violated, find customers to move.
    
    # To improve on greedy, we can try different initial assignments or different 
    # ways to handle the open cost trade-off.
    
    # Strategy: 
    # Run a local search where we try to swap assignments or move customers 
    # to reduce total cost (open + assign).
    
    # Given the small size (100x20), we can afford more sophisticated search.
    # Let's try a simple "Swap" local search on the greedy seed.
    
    # 1. Start with greedy result.
    current_result = best_result.copy()
    current_load = np.zeros(n_fac)
    for i in range(n_cust):
        current_load[current_result[i]] += demands[i]
        
    # Calculate current cost
    def calc_cost(res, load):
        used = np.unique(res)
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])

    current_cost = calc_cost(current_result, current_load)
    
    # Local search: 
    # Iterate until no improvement or time limit.
    # Move 1: Move a single customer to a different facility.
    # Move 2: Swap two customers.
    
    # To make it faster, we'll only consider moves that might improve cost.
    # Improving a single customer move:
    # Gain = (open_cost_change) + (assign_cost_new - assign_cost_old)
    # open_cost_change is 0 if facility is already used, -open_cost[f_old] if f_old becomes empty, +open_cost[f_new] if f_new was empty.
    
    # This is complex to vectorize perfectly for the open cost part because 
    # "becomes empty" depends on the specific customer.
    
    # Alternative: Just use a randomized restart with the "Effective Cost" heuristic.
    # The greedy is deterministic. Let's try a heuristic that penalizes opening facilities 
    # more or less.
    
    # Heuristic:
    # For each customer, choose facility f that minimizes assign_cost[i, f] + alpha * open_cost[f] / capacities[f]
    # Then repair capacity violations by moving customers to nearest feasible facility.
    
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for alpha in alphas:
        if time.time() > deadline - 5.0:
            break
            
        # 1. Initial Assignment based on effective cost
        # eff_cost[i, f] = assign_cost[i, f] + alpha * open_cost[f]
        # This doesn't account for capacity or sharing, but it biases towards cheap facilities.
        eff_cost = assign_cost + alpha * open_cost
        init_res = np.argmin(eff_cost, axis=1)
        
        # 2. Repair Capacity Violations
        # We need to ensure load[f] <= capacities[f].
        # If violated, move customers from overloaded facilities to underloaded ones.
        # Move the customer with the largest "loss" (increase in assign cost) first? 
        # Or smallest loss? To minimize total cost increase, we should move the one 
        # with the smallest penalty for moving away.
        
        res = init_res.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]
            
        # Check for violations
        # We'll iterate until no violations or max iterations.
        # Simple repair:
        # While any facility is overloaded:
        #   Pick an overloaded facility f.
        #   Pick a customer i in f who can be moved to a feasible facility g 
        #       with the minimum increase in assign_cost.
        #   Move i to g.
        
        max_iter = 100
        for _ in range(max_iter):
            if time.time() > deadline - 2.0:
                break
                
            overloaded = np.where(load > capacities)[0]
            if len(overloaded) == 0:
                break
                
            f = overloaded[0]
            # Find customers in f
            custs_in_f = np.where(res == f)[0]
            if len(custs_in_f) == 0:
                # Should not happen if load[f] > capacities[f] and demands > 0
                break
                
            best_cust = -1
            best_g = -1
            min_penalty = np.inf
            
            for i in custs_in_f:
                # Try moving i to any other facility g
                # g must have capacity: load[g] + demands[i] <= capacities[g]
                # and g != f
                for g in range(n_fac):
                    if g == f:
                        continue
                    if load[g] + demands[i] <= capacities[g]:
                        # Penalty is increase in assign cost
                        # Note: Moving i might open g (if load[g] was 0) or close f (if i was last).
                        # For simplicity in repair, we just look at assign cost delta.
                        # The open cost effect is handled by the initial heuristic or subsequent local search.
                        penalty = assign_cost[i, g] - assign_cost[i, f]
                        if penalty < min_penalty:
                            min_penalty = penalty
                            best_cust = i
                            best_g = g
            
            if best_cust == -1:
                # No feasible move found? This implies infeasibility or we need to swap.
                # For now, break. The result might be invalid.
                # We should verify validity before returning.
                break
                
            # Perform move
            res[best_cust] = best_g
            load[f] -= demands[best_cust]
            load[best_g] += demands[best_cust]
            
        # Verify validity
        if np.all(load <= capacities):
            # Calculate score
            used = np.unique(res)
            score = np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
            if score < best_score:
                best_score = score
                best_result = res.copy()
                
    # One more pass: Local Search on the best result found so far.
    # Try to swap or move customers to reduce cost.
    # This is O(N^2 * F) per iteration. 100^2 * 20 = 200,000 ops. Very fast.
    # We can run this for a few seconds.
    
    current_res = best_result.copy()
    current_load = np.zeros(n_fac)
    for i in range(n_cust):
        current_load[current_res[i]] += demands[i]
    
    # Precompute current cost components
    # cost_assign = sum(assign_cost[i, res[i]])
    # cost_open = sum(open_cost[f] for f in used)
    
    # To speed up local search, we can track the marginal cost of moving.
    # But let's just do a simple hill climbing with random restarts if time permits.
    # Given the budget, let's do a deterministic local search first.
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 1.0:
            break
            
        # Try moving each customer to every other facility
        # To avoid O(N^2 F) python loops, we can vectorize the "best move" search?
        # Hard due to capacity constraints and open cost dependencies.
        # But N=100, F=20. 2000 moves to check.
        # For each move, check validity and cost delta.
        
        for i in range(n_cust):
            if time.time() > deadline - 1.0:
                break
                
            f_old = current_res[i]
            # Try moving i to f_new
            # Cost delta:
            # 1. Assign cost change: assign_cost[i, f_new] - assign_cost[i, f_old]
            # 2. Open cost change:
            #    - If current_load[f_old] == demands[i] (i is last in f_old), f_old closes: -open_cost[f_old]
            #    - If current_load[f_new] == 0 (i is first in f_new), f_new opens: +open_cost[f_new]
            #    - Otherwise 0.
            
            # We need to find the f_new that minimizes this delta, subject to capacity.
            
            best_delta = np.inf
            best_f_new = -1
            
            for f_new in range(n_fac):
                if f_new == f_old:
                    continue
                if current_load[f_new] + demands[i] > capacities[f_new]:
                    continue
                    
                delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                
                delta_open = 0.0
                if current_load[f_old] == demands[i]:
                    delta_open -= open_cost[f_old]
                if current_load[f_new] == 0:
                    delta_open += open_cost[f_new]
                    
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_f_new = f_new
            
            if best_f_new != -1 and best_delta < -1e-9: # Strict improvement
                # Perform move
                current_res[i] = best_f_new
                current_load[f_old] -= demands[i]
                current_load[best_f_new] += demands[i]
                improved = True
                # Restart loop? Or just continue? 
                # Continuing is okay for a single pass, but restarting ensures we 
                # don't miss better moves enabled by this one.
                # However, restarting is O(N) overhead. With N=100, it's fine.
                # Let's just break out of inner loop and restart outer?
                # Actually, a simple single pass is often enough for a "kick".
                # Let's just mark improved and continue checking others.
                # It might not be optimal, but it's fast.
                
    # Final check
    if np.all(current_load <= capacities):
        used = np.unique(current_res)
        final_score = np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), current_res])
        if final_score < best_score:
            best_result = current_res
            best_score = final_score

    return best_result
# EVOLVE-BLOCK-END

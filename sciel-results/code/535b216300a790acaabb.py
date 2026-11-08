"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute sorted facility indices for each customer (cheapest first)
    sorted_fac_by_cost = np.argsort(assign_cost, axis=1)
    
    def calculate_cost(assignment):
        total_assign_cost = np.sum(assign_cost[np.arange(n_cust), assignment])
        used_facilities = np.unique(assignment)
        total_open_cost = np.sum(open_cost[used_facilities])
        return total_assign_cost + total_open_cost

    def is_valid(assignment):
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[assignment[i]] += demands[i]
        if np.any(loads > capacities):
            return False
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return False
        return True

    def greedy_assign(order, dual_prices=None):
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        # If dual prices are provided, adjust the effective cost
        if dual_prices is not None:
            adjusted_cost = assign_cost + dual_prices[np.newaxis, :]
            # Re-sort based on adjusted cost
            adj_sorted = np.argsort(adjusted_cost, axis=1)
        else:
            adj_sorted = sorted_fac_by_cost
            
        for i in order:
            d_i = demands[i]
            # Try facilities in order of (potentially adjusted) cost
            for f in adj_sorted[i]:
                if load[f] + d_i <= capacities[f]:
                    result[i] = f
                    load[f] += d_i
                    break
            if result[i] == -1:
                # Fallback: pick facility with most remaining capacity
                best_f = -1
                max_rem = -1
                for f in range(n_fac):
                    rem = capacities[f] - load[f]
                    if rem >= d_i and rem > max_rem:
                        max_rem = rem
                        best_f = f
                if best_f == -1:
                    return None
                result[i] = best_f
                load[best_f] += d_i
        return result

    def local_search(initial_assignment, max_iters=500):
        """
        Perform local search starting from initial_assignment.
        Uses Relocate and Swap moves.
        Returns the best assignment found and its cost.
        """
        current_assignment = initial_assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        # Precompute loads
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[current_assignment[i]] += demands[i]
            
        # Track customer count per facility for fast open cost delta calculation
        cust_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            cust_count[current_assignment[i]] += 1

        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            # 1. Relocate moves
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                f_from = current_assignment[i]
                # Try all other facilities sorted by cost
                for f_to in sorted_fac_by_cost[i]:
                    if f_to == f_from:
                        continue
                    # Check capacity
                    if load[f_to] + demands[i] > capacities[f_to]:
                        continue
                    
                    # Calculate cost change
                    delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                    
                    # Open cost change
                    delta_open = 0.0
                    # If f_from becomes empty (cust_count[f_from] == 1)
                    if cust_count[f_from] == 1:
                        delta_open -= open_cost[f_from]
                    # If f_to was empty (cust_count[f_to] == 0)
                    if cust_count[f_to] == 0:
                        delta_open += open_cost[f_to]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        # Accept move
                        current_assignment[i] = f_to
                        load[f_from] -= demands[i]
                        load[f_to] += demands[i]
                        cust_count[f_from] -= 1
                        cust_count[f_to] += 1
                        current_cost += total_delta
                        improved = True
                        break # Move to next customer

            # 2. Swap moves
            # Iterate over pairs of used facilities
            used_facs = [f for f in range(n_fac) if cust_count[f] > 0]
            
            found_swap = False
            for idx1, f1 in enumerate(used_facs):
                if time.time() > deadline or found_swap:
                    break
                for f2 in used_facs[idx1+1:]:
                    if time.time() > deadline or found_swap:
                        break
                    
                    # Get customers in f1 and f2
                    # Optimization: Only check a subset if lists are large? 
                    # For n=100, full check is okay.
                    custs_f1 = [i for i in range(n_cust) if current_assignment[i] == f1]
                    custs_f2 = [i for i in range(n_cust) if current_assignment[i] == f2]
                    
                    for i in custs_f1:
                        if found_swap:
                            break
                        for j in custs_f2:
                            # Check capacity after swap
                            # i goes to f2, j goes to f1
                            # load[f2] - demands[j] + demands[i] <= capacities[f2]
                            # load[f1] - demands[i] + demands[j] <= capacities[f1]
                            
                            if load[f2] - demands[j] + demands[i] > capacities[f2]:
                                continue
                            if load[f1] - demands[i] + demands[j] > capacities[f1]:
                                continue
                            
                            # Cost change
                            delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                            
                            # Open cost change is 0 for swaps between non-empty facilities
                            if delta_assign < -1e-9:
                                # Accept swap
                                current_assignment[i] = f2
                                current_assignment[j] = f1
                                load[f1] -= demands[i]
                                load[f1] += demands[j]
                                load[f2] -= demands[j]
                                load[f2] += demands[i]
                                # cust_count stays the same
                                current_cost += delta_assign
                                improved = True
                                found_swap = True
                                break
            if found_swap:
                # Restart the loop to re-evaluate other moves? 
                # Usually local search continues. The 'improved' flag handles it.
                pass

        return current_assignment, current_cost

    # 1. Generate diverse initial solutions
    candidates = []
    
    # Strategy 1: Largest demand first
    order1 = np.argsort(-demands)
    cand1 = greedy_assign(order1)
    if cand1 is not None and is_valid(cand1):
        candidates.append(cand1)
        
    # Strategy 2: Smallest demand first
    order2 = np.argsort(demands)
    cand2 = greedy_assign(order2)
    if cand2 is not None and is_valid(cand2):
        candidates.append(cand2)
        
    # Strategy 3: Random orders
    rng = np.random.default_rng(42)
    for _ in range(10):
        order3 = rng.permutation(n_cust)
        cand3 = greedy_assign(order3)
        if cand3 is not None and is_valid(cand3):
            candidates.append(cand3)
            
    # Strategy 4: Cheapest edge first (sort by min assign cost)
    min_costs = np.min(assign_cost, axis=1)
    order4 = np.argsort(min_costs)
    cand4 = greedy_assign(order4)
    if cand4 is not None and is_valid(cand4):
        candidates.append(cand4)

    if not candidates:
        # Emergency fallback
        return greedy_assign(np.argsort(-demands))

    # Select best candidate
    best_cand = min(candidates, key=calculate_cost)
    
    # Run local search on the best candidate
    best_assignment, best_cost = local_search(best_cand)
    
    # 2. Lagrangian Relaxation / Subgradient Optimization
    # We try to improve the solution by adjusting dual prices to guide the greedy assignment
    # and then applying local search to the result.
    
    # Initialize dual prices (congestion prices)
    dual_prices = np.zeros(n_fac)
    
    # We will run a few iterations of subgradient method
    # Step size parameter
    step_size = 10.0
    
    # Keep track of best solution found so far (from initial local search)
    global_best_assignment = best_assignment.copy()
    global_best_cost = best_cost
    
    # Limit number of Lagrangian iterations to avoid timeout
    max_lag_iters = 20
    
    for k in range(max_lag_iters):
        if time.time() > deadline:
            break
            
        # Generate a new solution using the current dual prices
        # We use the 'Largest Demand First' order as a base, but dual prices guide the facility choice
        # Note: greedy_assign with dual_prices sorts by assign_cost + dual_prices
        # However, simple greedy with adjusted costs might still get stuck.
        # Let's try a few different orders with the adjusted costs.
        
        # Order 1: Largest demand
        order_lag1 = np.argsort(-demands)
        cand_lag1 = greedy_assign(order_lag1, dual_prices)
        
        # Order 2: Random
        order_lag2 = rng.permutation(n_cust)
        cand_lag2 = greedy_assign(order_lag2, dual_prices)
        
        current_lag_candidates = [c for c in [cand_lag1, cand_lag2] if c is not None and is_valid(c)]
        
        if current_lag_candidates:
            # Pick the best among these Lagrangian-guided candidates
            best_lag_cand = min(current_lag_candidates, key=calculate_cost)
            
            # Refine with local search
            refined_assignment, refined_cost = local_search(best_lag_cand, max_iters=100) # Fewer iterations to save time
            
            if refined_cost < global_best_cost:
                global_best_assignment = refined_assignment
                global_best_cost = refined_cost
                
            # Calculate subgradient
            # The subgradient for facility f is (load_f - capacity_f) if load_f > 0?
            # Actually, for the constraint sum(d_i * x_if) <= cap_f, the subgradient is (load_f - cap_f).
            # If load_f > cap_f, it's infeasible, but our greedy ensures feasibility.
            # Wait, the Lagrangian relaxation usually relaxes constraints.
            # Here, we are using dual prices to *penalize* crowded facilities in the cost function.
            # The standard subgradient update for minimizing the Lagrangian dual (maximizing the lower bound) 
            # is different. But here we are using it heuristically to diversify the search.
            
            # Let's use a simple congestion-based update.
            # If a facility is heavily used (close to capacity), increase its price.
            # If a facility is underused, decrease its price.
            
            loads = np.zeros(n_fac)
            for i in range(n_cust):
                loads[global_best_assignment[i]] += demands[i]
                
            # Subgradient direction: load - capacity
            # If load > capacity (shouldn't happen), push up.
            # If load < capacity, push down?
            # Let's use: dual_price += step * (load/capacity - 1)
            # This increases price for full facilities and decreases for empty ones.
            
            # Normalize capacities to avoid division by zero
            safe_caps = np.where(capacities > 0, capacities, 1)
            
            utilization = loads / safe_caps
            subgrad = utilization - 1.0
            
            # Update dual prices
            # Decay step size
            step_size *= 0.8
            dual_prices += step_size * subgrad
            
            # Keep dual prices non-negative? Or allow negative to encourage underused facilities?
            # Negative prices would mean "reward" for using that facility.
            # Let's clamp to a reasonable range, e.g., [-100, 100]
            dual_prices = np.clip(dual_prices, -100, 100)

    # Final check
    if not is_valid(global_best_assignment):
        # Fallback to the initial best
        if is_valid(best_assignment):
            return best_assignment
        return greedy_assign(np.argsort(-demands))

    return global_best_assignment

# EVOLVE-BLOCK-END

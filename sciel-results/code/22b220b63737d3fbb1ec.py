# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    Strategy: Lagrangian Relaxation for facility opening decisions, 
    combined with Vectorized 1-Exchange Local Search for assignments.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (150s limit)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute sorted facility indices by assignment cost for each customer
    # This helps in finding feasible moves quickly
    sorted_fac_indices = np.argsort(assign_cost, axis=1)
    
    def _total_cost(result):
        """Compute total cost: open cost + assignment cost"""
        used = np.zeros(n_fac, dtype=bool)
        for f in result:
            used[f] = True
        oc = np.sum(open_cost[used])
        ac = np.sum(assign_cost[np.arange(n_cust), result])
        return oc + ac
    
    def _is_feasible(result):
        """Check if assignment is feasible (no capacity violations)"""
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        return np.all(load <= capacities + 1e-9)

    def _solve_subproblem(lam, max_iters=2000, time_limit=None):
        """
        Solve the assignment sub-problem given Lagrange multipliers lam.
        This is a capacitated assignment problem where facility f has cost 
        assign_cost[i,f] + lam[f] for customer i.
        We use a vectorized local search (1-exchange) to find a good solution.
        """
        # Initialize with greedy assignment based on adjusted costs
        adjusted_cost = assign_cost + lam[None, :]
        
        # Greedy init: sort customers by demand (largest first) to handle capacity early
        order = np.argsort(-demands)
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        # We need to be careful with greedy: pick cheapest feasible facility
        # For speed, we do a simple pass. If it fails, we fallback to a more robust init.
        # Since n=100, n_f=20, this is small.
        
        # Better init: Use the sorted indices
        # We will use a "best fit" greedy
        for i in order:
            di = demands[i]
            # Get facilities sorted by adjusted cost for this customer
            facs = sorted_fac_indices[i]
            placed = False
            for f in facs:
                if load[f] + di <= capacities[f] + 1e-9:
                    load[f] += di
                    result[i] = f
                    placed = True
                    break
            if not placed:
                # If greedy fails, we need a different init strategy
                # Try random restarts or a different order
                return None, None
        
        if not _is_feasible(result):
            return None, None

        # Local Search: 1-Exchange (Move)
        # We want to minimize sum(assign_cost[i, result[i]]) + sum(lam[result[i]])
        # Note: The objective here is just the assignment part + penalties.
        # The "open cost" is handled by the Lagrangian update outside.
        
        # Precompute loads
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
            
        # Vectorized 1-move local search
        # For each customer i, check if moving to facility j reduces cost.
        # Delta cost = (assign_cost[i, j] + lam[j]) - (assign_cost[i, result[i]] + lam[result[i]])
        # Constraint: load[j] + demands[i] <= capacities[j]
        
        improved = True
        while improved and (time_limit is None or time.time() < time_limit):
            improved = False
            
            # Vectorize over customers? 
            # For each i, we need to find min over j of (cost[i,j] + lam[j]) subject to capacity.
            # This is hard to fully vectorize with dynamic capacity constraints.
            # However, we can vectorize the cost calculation and filter by capacity.
            
            # Let's iterate customers, but use numpy for the inner loop
            for i in range(n_cust):
                if time_limit is not None and time.time() > time_limit:
                    break
                
                current_f = result[i]
                current_cost_val = assign_cost[i, current_f] + lam[current_f]
                di = demands[i]
                
                # Calculate delta for all other facilities
                # new_cost_val = assign_cost[i, :] + lam
                # We want new_cost_val < current_cost_val
                # And load[j] + di <= capacities[j]
                
                feasible_mask = (load + di) <= capacities + 1e-9
                feasible_mask[current_f] = False # Can't move to same facility
                
                if not np.any(feasible_mask):
                    continue
                
                # Get costs for feasible facilities
                # We need the minimum cost among feasible ones
                # Since n_fac is small (20), we can just compute all and mask
                costs = assign_cost[i, :] + lam
                costs[~feasible_mask] = np.inf
                
                min_idx = np.argmin(costs)
                min_cost_val = costs[min_idx]
                
                if min_cost_val < current_cost_val - 1e-9:
                    # Move i from current_f to min_idx
                    load[current_f] -= di
                    load[min_idx] += di
                    result[i] = min_idx
                    improved = True
        
        # Calculate the "true" cost of this assignment (without Lagrange multipliers)
        # But for the Lagrangian method, we return the assignment and the 
        # "penalized" cost is not what we track as the final solution quality.
        # We track the actual cost.
        actual_cost = _total_cost(result)
        return result, actual_cost

    def _subgradient_update(lam, result, step_size):
        """
        Update Lagrange multipliers based on capacity violations.
        g_f = load_f - capacity_f
        If load > capacity, increase lam (penalize usage).
        If load < capacity, decrease lam (encourage usage, but not below 0).
        """
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        
        violations = load - capacities
        # Projected subgradient: lam_new = max(0, lam + step_size * violations)
        new_lam = np.maximum(0, lam + step_size * violations)
        return new_lam, load

    # Main Loop: Subgradient Optimization
    # Initialize lam to 0
    lam = np.zeros(n_fac)
    
    best_result = None
    best_cost = float('inf')
    
    # Initial solution: Greedy with original costs
    # We can use _solve_subproblem with lam=0
    res, cost = _solve_subproblem(lam, max_iters=100, time_limit=start_time + 5)
    if res is not None:
        best_result = res
        best_cost = cost
    
    # Subgradient parameters
    # Standard step size: step_k = alpha_k * (UB - LB) / ||g||^2
    # We don't have a good LB easily. We can use a fixed decreasing step size.
    # Or use "best-so-far" step size rule.
    
    # Let's use a simple decreasing step size
    # Start with a large step, decrease over time.
    # The cost scale is roughly 100 * (50..150) + 100 * (0..100*sqrt(2)) ~ 5000-15000 + 1000-14000 ~ 10k-30k
    # Violations can be up to 35 (max demand).
    # So step size should be in the range of cost/violation ~ 1000.
    
    base_step = 10.0
    max_iters = 1000 # Safety cap
    
    for k in range(max_iters):
        if time.time() > deadline - 10:
            break
        
        # Solve subproblem
        # Time limit for subproblem: give it most of the remaining time, but ensure we have time for update
        sub_time_limit = deadline - 5 - (time.time() - start_time) * 0.0 # This logic is tricky
        # Better: allocate fixed time per subproblem, or adaptive.
        # Let's say 0.1s per subproblem is enough for 100x20?
        # 100 customers * 20 facilities = 2000 checks per pass. 10 passes = 20k ops. Very fast.
        # So we can run many iterations.
        
        res, cost = _solve_subproblem(lam, max_iters=100, time_limit=time.time() + 0.5)
        
        if res is None:
            # If subproblem infeasible with current lam, try to repair or skip
            # This shouldn't happen often if lam is high enough
            continue
            
        # Update best
        if cost < best_cost:
            best_cost = cost
            best_result = res.copy()
        
        # Update Lagrange multipliers
        # Step size rule:
        # step = alpha * (best_cost - current_penalized_cost) / (sum(violations^2) + eps)
        # current_penalized_cost = sum(assign_cost[i, res[i]] + lam[res[i]])
        
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]
        
        violations = load - capacities
        penalty_term = np.sum(violations**2) + 1e-6
        
        # Penalized cost (Lagrangian value)
        lagrangian_cost = np.sum(assign_cost[np.arange(n_cust), res]) + np.sum(lam[res])
        
        # Gap approximation
        gap = best_cost - lagrangian_cost
        if gap < 0:
            gap = 0.0
            
        # Step size
        alpha = 1.0 # Scaling factor
        step = alpha * gap / penalty_term
        
        # Dampening: reduce step size over time
        # step = step * (1.0 / (1.0 + 0.01 * k))
        
        # If step is too large, it might oscillate. Cap it.
        if step > 50.0:
            step = 50.0
        if step < 0.1:
            step = 0.1
            
        new_lam, _ = _subgradient_update(lam, res, step)
        lam = new_lam
        
        # Early stopping if no improvement for a while?
        # Subgradient methods are slow to converge.
        # We rely on the time limit.
        
    # Final Polish: 
    # The solution found by Lagrangian relaxation might not be the absolute best 
    # in the neighborhood of the best found so far.
    # Run a final, more aggressive local search on best_result.
    
    if best_result is not None:
        # Use the local search from the original code but optimized
        # We can reuse the logic from _solve_subproblem but with lam=0 
        # and more iterations.
        
        # Actually, _solve_subproblem with lam=0 IS the local search.
        # Let's run it again on best_result? No, it re-initializes.
        
        # Let's implement a dedicated polish pass on best_result
        res = best_result.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]
        
        # 1-Exchange Polish
        improved = True
        polish_deadline = deadline - 2
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(n_cust):
                if time.time() > polish_deadline:
                    break
                current_f = res[i]
                di = demands[i]
                current_cost_val = assign_cost[i, current_f]
                
                feasible_mask = (load + di) <= capacities + 1e-9
                feasible_mask[current_f] = False
                
                if not np.any(feasible_mask):
                    continue
                
                costs = assign_cost[i, :]
                costs[~feasible_mask] = np.inf
                
                min_idx = np.argmin(costs)
                min_cost_val = costs[min_idx]
                
                # Check open cost impact
                # If moving to min_idx opens a new facility, add open_cost[min_idx]
                # If moving from current_f closes it, subtract open_cost[current_f]
                
                # We need to track used facilities
                # This is slightly complex in vectorized form.
                # Let's do a simple check:
                
                # If min_idx is not in current load (load[min_idx]==0 before move? No, load is current)
                # We need to know if min_idx was used.
                # used_mask = load > 0 ? Not quite, if load becomes 0 after move, it was used.
                
                # Let's compute delta open cost
                # Current state:
                # load[current_f] >= di
                # load[min_idx] < capacities[min_idx] - di (feasible)
                
                # New state:
                # new_load[current_f] = load[current_f] - di
                # new_load[min_idx] = load[min_idx] + di
                
                delta_open = 0
                # If current_f becomes empty
                if load[current_f] - di < 1e-9:
                    delta_open -= open_cost[current_f]
                # If min_idx was empty
                if load[min_idx] < 1e-9:
                    delta_open += open_cost[min_idx]
                
                total_delta = (min_cost_val - current_cost_val) + delta_open
                
                if total_delta < -1e-9:
                    load[current_f] -= di
                    load[min_idx] += di
                    res[i] = min_idx
                    improved = True
        
        # Final check
        if _is_feasible(res):
            final_cost = _total_cost(res)
            if final_cost < best_cost:
                best_result = res
                best_cost = final_cost

    # Fallback if no solution found (should not happen)
    if best_result is None:
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            f = i % n_fac
            while load[f] + demands[i] > capacities[f]:
                f = (f + 1) % n_fac
            load[f] += demands[i]
            best_result[i] = f

    return best_result.astype(int)

# EVOLVE-BLOCK-END

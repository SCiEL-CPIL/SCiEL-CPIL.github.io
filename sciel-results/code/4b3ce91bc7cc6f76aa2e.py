"""Capacitated facility location: Lagrangian-guided greedy + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities minimizing open + assignment cost under capacity."""
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill

    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    def total_cost(assignment):
        """Compute total cost for a given assignment."""
        cost = np.sum(assign_cost[np.arange(n_cust), assignment])
        used = np.zeros(n_fac, dtype=bool)
        for f in assignment:
            used[f] = True
        cost += np.sum(open_cost[used])
        return cost

    def is_valid(assignment):
        """Check if assignment is valid (within capacity)."""
        load = np.zeros(n_fac, dtype=float)
        for c in range(n_cust):
            load[assignment[c]] += demands[c]
        return np.all(load <= capacities + 1e-9)

    def greedy_with_prices(prices):
        """Greedy assignment using effective cost = assign_cost + price*capacity_usage.
        Prices guide which facilities are 'expensive' to use, helping balance load."""
        # Effective cost for customer c at facility f: assign_cost[c,f] + prices[f] * demands[c]
        eff_cost = assign_cost + prices[None, :] * demands[:, None]
        
        assignment = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac, dtype=float)
        
        # Sort by demand descending (largest first)
        order = np.argsort(-demands)
        
        for c in order:
            # Find cheapest feasible facility
            best_f = -1
            best_cost = np.inf
            for f in np.argsort(eff_cost[c]):
                if load[f] + demands[c] <= capacities[f] + 1e-9:
                    best_f = f
                    best_cost = eff_cost[c, f]
                    break
            if best_f == -1:
                # No feasible facility; pick one with most remaining capacity
                remaining = capacities - load
                best_f = int(np.argmax(remaining))
                if load[best_f] + demands[c] > capacities[best_f] + 1e-9:
                    # Force assign to least-loaded facility (may violate, but we'll fix)
                    best_f = int(np.argmin(load))
            load[best_f] += demands[c]
            assignment[c] = best_f
        
        return assignment

    def local_search(assignment, max_iters=5000):
        """Improve assignment via single-customer reassignment."""
        assignment = assignment.copy()
        load = np.zeros(n_fac, dtype=float)
        for c in range(n_cust):
            load[assignment[c]] += demands[c]
        
        improved = True
        iters = 0
        best_cost = total_cost(assignment)
        
        while improved and iters < max_iters:
            if time.time() - start_time > deadline - 5:
                break
            improved = False
            iters += 1
            
            for c in range(n_cust):
                cur_f = assignment[c]
                cur_load = load[cur_f]
                
                # Try moving customer c to another facility
                for f in range(n_fac):
                    if f == cur_f:
                        continue
                    if load[f] + demands[c] > capacities[f] + 1e-9:
                        continue
                    
                    # Delta cost: (assign_cost[c,f] - assign_cost[c,cur_f]) + open_cost delta
                    delta_assign = assign_cost[c, f] - assign_cost[c, cur_f]
                    
                    # Open cost delta: if cur_f becomes empty and f was empty
                    old_open = 0.0
                    new_open = 0.0
                    if cur_load == demands[c]:  # cur_f would become empty
                        old_open = -open_cost[cur_f]
                    if load[f] == 0:  # f would become used
                        new_open = open_cost[f]
                    
                    delta = delta_assign + old_open + new_open
                    
                    if delta < -1e-9:
                        # Make the move
                        load[cur_f] -= demands[c]
                        load[f] += demands[c]
                        assignment[c] = f
                        best_cost += delta
                        improved = True
                        break  # Restart scan from customer 0
        
        return assignment, best_cost

    def subgradient_update(assignment, prices, step_size, n_iter):
        """Update dual prices via subgradient method on Lagrangian relaxation.
        The subgradient for price[f] is (load[f] - capacities[f]) if we relax capacity.
        Actually, for the standard CFLP Lagrangian, we relax capacity constraints:
        L = sum_c sum_f x_cf * assign_cost_cf + sum_f y_f * open_cost_f
            + sum_f lambda_f * (sum_c x_cf * demands_c - capacities_f)
        Subgradient for lambda_f: sum_c x_cf * demands_c - capacities_f = load_f - cap_f
        """
        load = np.zeros(n_fac, dtype=float)
        for c in range(n_cust):
            load[assignment[c]] += demands[c]
        
        # Subgradient: load_f - capacities_f (positive means over capacity)
        subgrad = load - capacities
        
        # Normalize
        norm = np.linalg.norm(subgrad)
        if norm < 1e-12:
            return prices
        
        prices = prices + step_size * subgrad / norm
        prices = np.maximum(prices, 0)  # Prices must be non-negative
        
        return prices

    # --- Phase 1: Get initial solution via greedy ---
    init_prices = np.zeros(n_fac)
    assignment = greedy_with_prices(init_prices)
    
    # Fix any infeasibility
    if not is_valid(assignment):
        # Reassign overloaded customers to feasible facilities
        load = np.zeros(n_fac, dtype=float)
        for c in range(n_cust):
            load[assignment[c]] += demands[c]
        
        for c in range(n_cust):
            if load[assignment[c]] > capacities[assignment[c]] + 1e-9:
                # Find a facility with capacity
                for f in np.argsort(assign_cost[c]):
                    if load[f] + demands[c] <= capacities[f] + 1e-9:
                        load[assignment[c]] -= demands[c]
                        load[f] += demands[c]
                        assignment[c] = f
                        break
    
    best_assignment = assignment.copy()
    best_cost = total_cost(assignment)
    
    # --- Phase 2: Subgradient optimization to find good prices ---
    n_subgrad_iters = 300
    best_subgrad_cost = np.inf
    best_subgrad_assignment = assignment.copy()
    
    for it in range(n_subgrad_iters):
        if time.time() - start_time > deadline - 10:
            break
        
        # Step size decreases
        step_size = 2.0 / (it + 1)
        
        # Greedy with current prices
        assignment = greedy_with_prices(prices)
        
        # Check validity and fix
        if not is_valid(assignment):
            load = np.zeros(n_fac, dtype=float)
            for c in range(n_cust):
                load[assignment[c]] += demands[c]
            for c in range(n_cust):
                if load[assignment[c]] > capacities[assignment[c]] + 1e-9:
                    for f in np.argsort(assign_cost[c]):
                        if load[f] + demands[c] <= capacities[f] + 1e-9:
                            load[assignment[c]] -= demands[c]
                            load[f] += demands[c]
                            assignment[c] = f
                            break
        
        cost = total_cost(assignment)
        if cost < best_subgrad_cost:
            best_subgrad_cost = cost
            best_subgrad_assignment = assignment.copy()
        
        # Update prices
        load = np.zeros(n_fac, dtype=float)
        for c in range(n_cust):
            load[assignment[c]] += demands[c]
        subgrad = load - capacities
        norm = np.linalg.norm(subgrad)
        if norm > 1e-12:
            prices = prices + step_size * subgrad / norm
            prices = np.maximum(prices, 0)
    
    if best_subgrad_assignment is not None and best_subgrad_cost < best_cost:
        best_assignment = best_subgrad_assignment
        best_cost = best_subgrad_cost
    
    # --- Phase 3: Local search refinement ---
    improved_assignment, improved_cost = local_search(best_assignment, max_iters=10000)
    if improved_cost < best_cost:
        best_assignment = improved_assignment
        best_cost = improved_cost
    
    # --- Phase 4: Multiple restarts with different price vectors if time permits ---
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - start_time > deadline - 15:
            break
        
        # Random prices
        prices_r = np.random.rand(n_fac) * 10.0
        assignment_r = greedy_with_prices(prices_r)
        
        # Fix infeasibility
        if not is_valid(assignment_r):
            load = np.zeros(n_fac, dtype=float)
            for c in range(n_cust):
                load[assignment_r[c]] += demands[c]
            for c in range(n_cust):
                if load[assignment_r[c]] > capacities[assignment_r[c]] + 1e-9:
                    for f in np.argsort(assign_cost[c]):
                        if load[f] + demands[c] <= capacities[f] + 1e-9:
                            load[assignment_r[c]] -= demands[c]
                            load[f] += demands[c]
                            assignment_r[c] = f
                            break
        
        # Local search
        improved_r, cost_r = local_search(assignment_r, max_iters=5000)
        if cost_r < best_cost:
            best_assignment = improved_r
            best_cost = cost_r
    
    # Final validation
    if not is_valid(best_assignment):
        # Last resort: use the simple greedy
        load = np.zeros(n_fac, dtype=float)
        for c in np.argsort(-demands):
            for f in np.argsort(assign_cost[c]):
                if load[f] + demands[c] <= capacities[f] + 1e-9:
                    load[f] += demands[c]
                    best_assignment[c] = f
                    break
    
    return best_assignment.astype(int)
# EVOLVE-BLOCK-END

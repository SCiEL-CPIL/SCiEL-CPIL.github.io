"""Lagrangian relaxation + subgradient + guided repair + swap local search for CFLP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Strategy: Lagrangian relaxation with subgradient optimization to get dual prices,
    then Lagrangian-guided heuristic repair, capacity repair, and swap-based local search.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    total_demand = demands.sum()

    def compute_cost(assignment):
        """Compute total cost for an assignment, return inf if infeasible."""
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(assignment[c])
            if f < 0 or f >= n_facilities:
                return np.inf
            load[f] += demands[c]
        for f in range(n_facilities):
            if load[f] > capacities[f] + 1e-9:
                return np.inf
        cost = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            f = int(assignment[c])
            cost += assign_cost[c, f]
            used[f] = True
        for f in range(n_facilities):
            if used[f]:
                cost += open_cost[f]
        return cost

    def lagrangian_lower_bound(mu):
        """
        Given dual multipliers mu for facility capacity constraints,
        compute the Lagrangian lower bound and the resulting (possibly infeasible) assignment.
        L(mu) = sum_c min_f (assign_cost[c,f] + mu_f) + sum_f (mu_f * cap_f - open_cost_f)
        where we treat the fixed charge as: if facility f is used, subtract open_cost_f.
        Actually the standard CFLP Lagrangian relaxation on capacity constraints:
        min sum_c sum_f a_cf x_cf + sum_f o_f y_f
        s.t. sum_c x_cf <= cap_f y_f, sum_f x_cf = 1, x_cf in {0,1}, y_f in {0,1}
        
        Relaxing capacity: L(mu) = sum_c min_f (a_cf + mu_f) + sum_f (mu_f * cap_f - o_f)
        (the y_f part: if mu_f > 0 and facility is "open" in the relaxed solution, we get -o_f)
        
        But more carefully: the dual of the capacity constraint gives us that if we relax
        sum_c x_cf <= cap_f, we add mu_f * (sum_c x_cf - cap_f) to the objective.
        Then for each customer c, we pick f minimizing a_cf + mu_f.
        The lower bound is: sum_c min_f (a_cf + mu_f) - sum_f mu_f * cap_f
        And we also need to account for the fixed charge: if facility f is selected by any customer,
        we add o_f. In the relaxed problem, y_f is determined by whether any customer picks f.
        So L(mu) = sum_c min_f (a_cf + mu_f) + sum_f o_f * I(any c picks f) - sum_f mu_f * cap_f
        """
        # For each customer, find best facility under modified costs
        modified = assign_cost + mu[None, :]  # shape (n_customers, n_facilities)
        best_f = np.argmin(modified, axis=1)
        best_val = modified[np.arange(n_customers), best_f]
        
        # Which facilities are used?
        used = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            used[best_f[c]] = True
        
        fixed_charge = open_cost[used].sum()
        bound = best_val.sum() + fixed_charge - mu @ capacities
        
        return bound, best_f, used

    def subgradient_optimize(max_iter=2000, initial_mu=None):
        """
        Subgradient optimization to find good dual multipliers.
        Returns (best_mu, best_bound, best_assignment, best_used).
        """
        if initial_mu is None:
            # Initialize mu based on capacity tightness
            mu = np.zeros(n_facilities)
            # Simple initialization: proportional to open_cost / capacity
            for f in range(n_facilities):
                if capacities[f] > 0:
                    mu[f] = open_cost[f] / capacities[f] * 0.1
                else:
                    mu[f] = 0.0
        else:
            mu = initial_mu.copy()
        
        best_bound = np.inf
        best_mu = mu.copy()
        best_assign = None
        best_used = None
        
        # Subgradient step size
        step_scale = 2.0  # multiplier for step size
        
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            bound, assign_f, used = lagrangian_lower_bound(mu)
            
            if bound < best_bound:
                best_bound = bound
                best_mu = mu.copy()
                best_assign = assign_f.copy()
                best_used = used.copy()
            
            # Subgradient: g_f = sum_c x_cf - cap_f (where x_cf = 1 if c assigned to f)
            load = np.zeros(n_facilities)
            for c in range(n_customers):
                load[assign_f[c]] += demands[c]
            grad = load - capacities  # positive if over capacity
            
            # Step size: Barzilai-Borwein style or simple
            # Simple: alpha = step_scale * (UB - LB) / ||g||^2
            # We don't have a tight UB, so use a heuristic
            # ||g||^2
            g_norm_sq = np.sum(grad ** 2)
            if g_norm_sq < 1e-12:
                break
            
            # Use a decaying step size
            alpha = step_scale * (1.0 + 0.5 * (1.0 - it / max(1, max_iter))) / np.sqrt(g_norm_sq)
            
            # Update mu: mu = max(0, mu - alpha * grad)
            mu = np.maximum(0.0, mu - alpha * grad)
        
        return best_mu, best_bound, best_assign, best_used

    def lagrangian_repair(mu, assignment_hint=None):
        """
        Given dual prices mu, create a feasible assignment.
        If assignment_hint is given, use it to break ties.
        Otherwise, assign each customer to argmin(assign_cost[c,f] + mu[f]).
        Then repair capacity violations.
        """
        if assignment_hint is not None:
            # Start from hint but re-optimize with Lagrangian guidance
            result = assignment_hint.copy()
        else:
            modified = assign_cost + mu[None, :]
            # Sort customers by demand (largest first for better capacity packing)
            order = np.argsort(-demands)
            result = np.full(n_customers, -1, dtype=int)
            load = np.zeros(n_facilities)
            for c in order:
                costs = modified[c].copy()
                feasible = load + demands[c] <= capacities
                costs[~feasible] = np.inf
                if np.all(np.isinf(costs)):
                    # No feasible facility, pick least infeasible
                    f = int(np.argmin(costs + load))
                else:
                    f = int(np.argmin(costs))
                result[c] = f
                load[f] += demands[c]
        
        # Repair capacity violations
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[result[c]] += demands[c]
        
        # Iteratively fix violations
        for _ in range(50):
            if time.time() > deadline - 20:
                break
            violations = [f for f in range(n_facilities) if load[f] > capacities[f] + 1e-9]
            if not violations:
                break
            
            for f in violations:
                # Find customers assigned to f, move the ones with highest marginal cost
                customers_at_f = [c for c in range(n_customers) if result[c] == f]
                if not customers_at_f:
                    continue
                
                # Sort by how much they "prefer" f (lower assign_cost means harder to move)
                # We want to move customers who have cheap alternatives
                def move_cost(c):
                    # Cost increase if we move c from f to best alternative
                    best_alt = np.inf
                    for f2 in range(n_facilities):
                        if f2 == f:
                            continue
                        if load[f2] + demands[c] <= capacities[f2] + 1e-9:
                            alt_cost = assign_cost[c, f2]
                            if load[f2] == 0:
                                alt_cost += open_cost[f2]
                            if alt_cost < best_alt:
                                best_alt = alt_cost
                    if best_alt == np.inf:
                        return np.inf
                    return best_alt - assign_cost[c, f]
                
                # Move customers with lowest move cost (easiest to relocate)
                customers_at_f.sort(key=move_cost)
                
                excess = load[f] - capacities[f]
                for c in customers_at_f:
                    if excess <= 1e-9:
                        break
                    mc = move_cost(c)
                    if mc == np.inf:
                        continue
                    # Move c to best alternative
                    best_alt_f = -1
                    best_alt_cost = np.inf
                    for f2 in range(n_facilities):
                        if f2 == f:
                            continue
                        if load[f2] + demands[c] <= capacities[f2] + 1e-9:
                            alt_cost = assign_cost[c, f2]
                            if load[f2] == 0:
                                alt_cost += open_cost[f2]
                            if alt_cost < best_alt_cost:
                                best_alt_cost = alt_cost
                                best_alt_f = f2
                    if best_alt_f >= 0:
                        load[f] -= demands[c]
                        load[best_alt_f] += demands[c]
                        result[c] = best_alt_f
                        excess -= demands[c]
        
        # Final check
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[result[c]] += demands[c]
        if any(load[f] > capacities[f] + 1e-9 for f in range(n_facilities)):
            return None
        return result

    def swap_local_search(assignment, max_iter=2000):
        """
        Local search with single reassignments and pairwise swaps.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment
        
        best_cost = current_cost
        best_assign = assignment.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            improved = False
            
            # Single reassignment
            for c in range(n_customers):
                old_f = int(assignment[c])
                load[old_f] -= demands[c]
                
                best_new_f = old_f
                best_delta = 0.0
                
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f] + 1e-9:
                        continue
                    
                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]
                    
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = new_f
                
                if best_new_f != old_f and best_delta < -1e-9:
                    assignment[c] = best_new_f
                    load[best_new_f] += demands[c]
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_assign = assignment.copy()
                else:
                    load[old_f] += demands[c]
            
            # Pairwise swaps
            for i in range(n_customers):
                for j in range(i + 1, n_customers):
                    fi = int(assignment[i])
                    fj = int(assignment[j])
                    if fi == fj:
                        continue
                    
                    # Swap: i -> fj, j -> fi
                    if load[fi] - demands[i] + demands[j] > capacities[fi] + 1e-9:
                        continue
                    if load[fj] - demands[j] + demands[i] > capacities[fj] + 1e-9:
                        continue
                    
                    delta = (assign_cost[i, fj] - assign_cost[i, fi]) + (assign_cost[j, fi] - assign_cost[j, fj])
                    
                    # Fixed charge changes
                    # After swap: fi has load - demands[i] + demands[j], fj has load - demands[j] + demands[i]
                    if load[fi] - demands[i] == 0 and load[fi] - demands[i] + demands[j] > 0:
                        delta += open_cost[fi]
                    if load[fi] - demands[i] > 0 and load[fi] - demands[i] + demands[j] == 0:
                        delta -= open_cost[fi]
                    if load[fj] - demands[j] == 0 and load[fj] - demands[j] + demands[i] > 0:
                        delta += open_cost[fj]
                    if load[fj] - demands[j] > 0 and load[fj] - demands[j] + demands[i] == 0:
                        delta -= open_cost[fj]
                    
                    if delta < -1e-9:
                        # Perform swap
                        load[fi] -= demands[i]
                        load[fj] -= demands[j]
                        load[fi] += demands[j]
                        load[fj] += demands[i]
                        assignment[i] = fj
                        assignment[j] = fi
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_assign = assignment.copy()
            
            if not improved:
                break
        
        return best_assign

    # === Main algorithm ===
    best_solution = None
    best_cost = np.inf
    
    # Phase 1: Subgradient optimization to get good dual prices
    if time.time() < deadline - 40:
        mu, lb, lag_assign, lag_used = subgradient_optimize(max_iter=3000)
        
        # Phase 2: Lagrangian-guided repair
        if time.time() < deadline - 30:
            # Repair using the Lagrangian assignment as hint
            sol = lagrangian_repair(mu, assignment_hint=lag_assign)
            if sol is not None:
                cost = compute_cost(sol)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = sol.copy()
            
            # Also try repair without hint (pure Lagrangian greedy)
            sol2 = lagrangian_repair(mu)
            if sol2 is not None:
                cost = compute_cost(sol2)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = sol2.copy()
        
        # Phase 3: Local search with swaps
        if best_solution is not None and time.time() < deadline - 15:
            improved = swap_local_search(best_solution, max_iter=3000)
            cost = compute_cost(improved)
            if cost < best_cost:
                best_cost = cost
                best_solution = improved
    
    # Phase 4: If no solution yet, fall back to greedy
    if best_solution is None:
        order = np.argsort(-demands)
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        for c in order:
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf
            if np.all(np.isinf(costs)):
                f = int(np.argmin(costs + load))
            else:
                f = int(np.argmin(costs))
            result[c] = f
            load[f] += demands[c]
        
        # Repair
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[result[c]] += demands[c]
        for _ in range(50):
            violations = [f for f in range(n_facilities) if load[f] > capacities[f] + 1e-9]
            if not violations:
                break
            for f in violations:
                customers_at_f = [c for c in range(n_customers) if result[c] == f]
                customers_at_f.sort(key=lambda c: assign_cost[c].min() - assign_cost[c, f])
                excess = load[f] - capacities[f]
                for c in customers_at_f:
                    if excess <= 1e-9:
                        break
                    best_alt_f = -1
                    best_alt_cost = np.inf
                    for f2 in range(n_facilities):
                        if f2 == f:
                            continue
                        if load[f2] + demands[c] <= capacities[f2] + 1e-9:
                            alt_cost = assign_cost[c, f2]
                            if alt_cost < best_alt_cost:
                                best_alt_cost = alt_cost
                                best_alt_f = f2
                    if best_alt_f >= 0:
                        load[f] -= demands[c]
                        load[best_alt_f] += demands[c]
                        result[c] = best_alt_f
                        excess -= demands[c]
        
        best_solution = result
        best_cost = compute_cost(best_solution)
        
        if best_solution is not None and time.time() < deadline - 10:
            improved = swap_local_search(best_solution, max_iter=2000)
            cost = compute_cost(improved)
            if cost < best_cost:
                best_cost = cost
                best_solution = improved
    
    if best_solution is None:
        best_solution = np.zeros(n_customers, dtype=int)
    
    return best_solution.astype(int)
# EVOLVE-BLOCK-END

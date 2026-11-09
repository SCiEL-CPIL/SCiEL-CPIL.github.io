"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def _solve_milp(demands, capacities, open_cost, assign_cost, time_limit=10):
    """
    Solve the capacitated facility location problem using MILP.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Variables:
    # x[c, f] for c in 0..n_cust-1, f in 0..n_fac-1 (n_cust * n_fac vars)
    # y[f] for f in 0..n_fac-1 (n_fac vars)
    n_x = n_cust * n_fac
    n_y = n_fac
    n_vars = n_x + n_y
    
    # Objective coefficients
    c = np.zeros(n_vars)
    c[:n_x] = assign_cost.flatten()
    c[n_x:] = open_cost
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_f x[c,f] = 1
    A_eq = np.zeros((n_cust, n_vars))
    for c_idx in range(n_cust):
        for f_idx in range(n_fac):
            A_eq[c_idx, c_idx * n_fac + f_idx] = 1.0
    b_eq = np.ones(n_cust)
    
    # 2. Capacity constraints: sum_c demands[c] * x[c,f] <= capacities[f] * y[f]
    A_ub = np.zeros((n_fac, n_vars))
    for f_idx in range(n_fac):
        for c_idx in range(n_cust):
            A_ub[f_idx, c_idx * n_fac + f_idx] = demands[c_idx]
        A_ub[f_idx, n_x + f_idx] = -capacities[f_idx]
    b_ub = np.zeros(n_fac)
    
    # Integrality: x and y are binary
    integrality = np.ones(n_vars)
    
    # Bounds: all variables between 0 and 1
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    constraint_eq = LinearConstraint(A_eq, lb=b_eq, ub=b_eq)
    constraint_ub = LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
    
    # Solve
    try:
        res = milp(c=c, constraints=[constraint_eq, constraint_ub], integrality=integrality, bounds=bounds, options={"time_limit": time_limit})
        
        if res.status == 0:
            x_flat = res.x[:n_x]
            x = np.round(x_flat).reshape(n_cust, n_fac)
            assignment = np.argmax(x, axis=1)
            
            # Check validity
            loads = np.zeros(n_fac)
            for i in range(n_cust):
                loads[assignment[i]] += demands[i]
            if np.any(loads > capacities + 1e-6):
                return None
            return assignment
        else:
            return None
    except Exception:
        return None

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    # Helper functions
    def calculate_cost(assignment):
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return float('inf')
        total_assign_cost = np.sum(assign_cost[np.arange(n_cust), assignment])
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[assignment[i]] += demands[i]
        if np.any(loads > capacities + 1e-9):
            return float('inf')
        used_facilities = np.unique(assignment)
        total_open_cost = np.sum(open_cost[used_facilities])
        return total_assign_cost + total_open_cost

    def is_valid(assignment):
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return False
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[assignment[i]] += demands[i]
        if np.any(loads > capacities + 1e-9):
            return False
        return True

    def greedy_assign(order, dual_prices=None):
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        if dual_prices is not None:
            adjusted_cost = assign_cost + dual_prices[np.newaxis, :]
            adj_sorted = np.argsort(adjusted_cost, axis=1)
        else:
            adj_sorted = np.argsort(assign_cost, axis=1)
            
        for i in order:
            d_i = demands[i]
            for f in adj_sorted[i]:
                if load[f] + d_i <= capacities[f] + 1e-9:
                    result[i] = f
                    load[f] += d_i
                    break
            if result[i] == -1:
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

    def local_search(initial_assignment, max_iters=200):
        current_assignment = initial_assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[current_assignment[i]] += demands[i]
            
        cust_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            cust_count[current_assignment[i]] += 1

        sorted_fac_by_cost = np.argsort(assign_cost, axis=1)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                f_from = current_assignment[i]
                for f_to in sorted_fac_by_cost[i]:
                    if f_to == f_from:
                        continue
                    if load[f_to] + demands[i] > capacities[f_to] + 1e-9:
                        continue
                    
                    delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                    delta_open = 0.0
                    if cust_count[f_from] == 1:
                        delta_open -= open_cost[f_from]
                    if cust_count[f_to] == 0:
                        delta_open += open_cost[f_to]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        current_assignment[i] = f_to
                        load[f_from] -= demands[i]
                        load[f_to] += demands[i]
                        cust_count[f_from] -= 1
                        cust_count[f_to] += 1
                        current_cost += total_delta
                        improved = True
                        break

        return current_assignment, current_cost

    def two_opt_swap(assignment, max_iters=50):
        """Try swapping pairs of customers between facilities."""
        current_assignment = assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[current_assignment[i]] += demands[i]
            
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
            
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                for j in range(i+1, n_cust):
                    if time.time() > deadline:
                        break
                    f_i = current_assignment[i]
                    f_j = current_assignment[j]
                    if f_i == f_j:
                        continue
                    
                    # Check if swap is feasible
                    if load[f_i] - demands[i] + demands[j] > capacities[f_i] + 1e-9:
                        continue
                    if load[f_j] - demands[j] + demands[i] > capacities[f_j] + 1e-9:
                        continue
                    
                    # Calculate delta
                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    total_delta = delta_assign
                    
                    if total_delta < -1e-9:
                        current_assignment[i] = f_j
                        current_assignment[j] = f_i
                        load[f_i] -= demands[i]
                        load[f_j] -= demands[j]
                        load[f_i] += demands[j]
                        load[f_j] += demands[i]
                        current_cost += total_delta
                        improved = True
                        break
                if improved:
                    continue

        return current_assignment, current_cost

    def lagrangian_relaxation(demands, capacities, open_cost, assign_cost, max_iters=100, deadline=None):
        """
        Lagrangian relaxation approach:
        - Dualize capacity constraints with multipliers lambda_f
        - For each iteration, solve the relaxed problem (assign each customer to min cost facility)
        - Update multipliers via subgradient method
        - Use the best feasible solution found during iterations
        """
        if deadline is None:
            deadline = time.time() + 50.0
        
        n_cust = len(demands)
        n_fac = len(capacities)
        
        # Initialize multipliers to zero
        lam = np.zeros(n_fac)
        
        best_assignment = None
        best_cost = float('inf')
        
        # Step size parameter
        alpha = 1.0
        best_ub = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Solve the Lagrangian subproblem:
            # For each customer c, choose facility f that minimizes assign_cost[c,f] + lam[f]
            # (ignoring capacity constraints)
            adjusted_cost = assign_cost + lam[np.newaxis, :]
            relaxed_assignment = np.argmin(adjusted_cost, axis=1)
            
            # Calculate the Lagrangian lower bound
            lagrangian_value = np.sum(assign_cost[np.arange(n_cust), relaxed_assignment]) + np.sum(lam * capacities)
            
            # Check if the relaxed assignment is feasible
            loads = np.zeros(n_fac)
            for i in range(n_cust):
                loads[relaxed_assignment[i]] += demands[i]
            
            if np.all(loads <= capacities + 1e-9):
                # Feasible! Calculate actual cost
                actual_cost = calculate_cost(relaxed_assignment)
                if actual_cost < best_cost:
                    best_cost = actual_cost
                    best_assignment = relaxed_assignment.copy()
                    best_ub = actual_cost
                # If we found a feasible solution, we can stop or continue
                # Let's continue a few more iterations to potentially find better
            else:
                # Infeasible: update multipliers via subgradient
                # Subgradient of the Lagrangian dual with respect to lam[f] is (load_f - capacity_f)
                subgrad = loads - capacities
                
                # Calculate step size
                if best_ub == float('inf'):
                    alpha = 1.0
                else:
                    # Use a diminishing step size
                    gap = best_ub - lagrangian_value
                    if gap > 1e-6:
                        alpha = 2.0 * gap / (np.sum(subgrad**2) + 1e-10)
                    else:
                        alpha = 0.1
                    
                    # Decay alpha over iterations
                    alpha *= (1.0 - 0.01 * it)
                    alpha = max(alpha, 0.01)
                
                # Update multipliers (project to non-negative)
                lam = np.maximum(lam + alpha * subgrad, 0.0)
                
                # Try to repair the infeasible solution to get a feasible one
                # Simple repair: move customers from overloaded facilities to least loaded
                repaired = relaxed_assignment.copy()
                rep_load = loads.copy()
                
                # Sort facilities by overload
                overloads = rep_load - capacities
                overloaded_facs = np.where(overloads > 1e-9)[0]
                
                for f_over in overloaded_facs:
                    # Move customers from this facility to others
                    customers_at_f = np.where(repaired == f_over)[0]
                    # Sort by increasing demand (move smaller ones first)
                    customers_at_f = customers_at_f[np.argsort(demands[customers_at_f])]
                    
                    for c in customers_at_f:
                        if rep_load[f_over] <= capacities[f_over] + 1e-9:
                            break
                        
                        d_c = demands[c]
                        # Find the best alternative facility
                        best_f = -1
                        best_delta = float('inf')
                        for f_to in range(n_fac):
                            if f_to == f_over:
                                continue
                            if rep_load[f_to] + d_c <= capacities[f_to] + 1e-9:
                                delta = assign_cost[c, f_to] - assign_cost[c, f_over]
                                if delta < best_delta:
                                    best_delta = delta
                                    best_f = f_to
                        
                        if best_f != -1:
                            repaired[c] = best_f
                            rep_load[f_over] -= d_c
                            rep_load[best_f] += d_c
                        else:
                            # Can't move this customer, skip
                            pass
                
                # Check if repair was successful
                if np.all(rep_load <= capacities + 1e-9):
                    actual_cost = calculate_cost(repaired)
                    if actual_cost < best_cost:
                        best_cost = actual_cost
                        best_assignment = repaired.copy()
                        best_ub = actual_cost
            
            # Occasionally try local search on the best solution found
            if best_assignment is not None and it % 10 == 0 and time.time() < deadline - 10:
                ls_result, ls_cost = local_search(best_assignment, max_iters=20)
                if ls_cost < best_cost:
                    best_cost = ls_cost
                    best_assignment = ls_result
        
        return best_assignment, best_cost

    # Strategy:
    # 1. Try MILP with short time limit
    # 2. Try Lagrangian relaxation
    # 3. Use local search and 2-opt to polish
    # 4. Return the best feasible solution found

    candidates = []
    
    # Attempt 1: MILP with short time
    milp_time = 8
    assignment_milp = _solve_milp(demands, capacities, open_cost, assign_cost, time_limit=milp_time)
    if assignment_milp is not None and is_valid(assignment_milp):
        cost_milp = calculate_cost(assignment_milp)
        candidates.append((cost_milp, assignment_milp))
    
    # Attempt 2: Lagrangian relaxation
    if time.time() < deadline - 30:
        lagr_assignment, lagr_cost = lagrangian_relaxation(
            demands, capacities, open_cost, assign_cost, 
            max_iters=80, deadline=deadline - 25
        )
        if lagr_assignment is not None and is_valid(lagr_assignment):
            if lagr_cost < best_cost or best_cost == float('inf'):
                pass
            if not any(np.array_equal(cand[1], lagr_assignment) for cand in candidates):
                candidates.append((calculate_cost(lagr_assignment), lagr_assignment))
    
    # Attempt 3: Greedy fallback if needed
    if len(candidates) == 0:
        order = np.argsort(-demands)
        greedy_result = greedy_assign(order)
        if greedy_result is not None and is_valid(greedy_result):
            candidates.append((calculate_cost(greedy_result), greedy_result))
        else:
            # Last resort
            return np.zeros(n_cust, dtype=int)
    
    # Select the best candidate
    candidates.sort(key=lambda x: x[0])
    best_assignment = candidates[0][1].copy()
    
    # Polish with local search
    if time.time() < deadline - 15:
        best_assignment, _ = local_search(best_assignment, max_iters=100)
    
    # Polish with 2-opt swap
    if time.time() < deadline - 10:
        best_assignment, _ = two_opt_swap(best_assignment, max_iters=30)
    
    # Final local search
    if time.time() < deadline - 5:
        best_assignment, _ = local_search(best_assignment, max_iters=50)
    
    if is_valid(best_assignment):
        return best_assignment
    else:
        # Fallback to any valid candidate
        for cost, assignment in candidates:
            if is_valid(assignment):
                return assignment
        return np.zeros(n_cust, dtype=int)

# EVOLVE-BLOCK-END

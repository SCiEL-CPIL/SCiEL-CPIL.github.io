"""Capacitated facility location: Lagrangian Relaxation with Subgradient Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian Relaxation with Subgradient Optimization.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    n_cust = len(demands)
    n_fac = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def _total_cost(result):
        load = np.zeros(n_fac)
        for c in range(n_cust):
            load[result[c]] += demands[c]
        used = load > 0
        open_sum = open_cost[used].sum()
        assign_sum = np.sum(assign_cost[np.arange(n_cust), result])
        return open_sum + assign_sum

    def _repair(result, load):
        """
        Greedy repair of an infeasible assignment.
        Moves customers from overloaded facilities to feasible ones.
        """
        result = result.copy()
        load = load.copy()
        count = np.zeros(n_fac, dtype=int)
        for c in range(n_cust):
            count[result[c]] += 1
            
        # Identify overloaded facilities
        overloaded = load > capacities + 1e-6
        
        # We need to move customers out of overloaded facilities
        # Sort customers in overloaded facilities by 'difficulty' (high demand, high cost to move)
        # Or simply iterate through overloaded facilities and move customers
        
        while np.any(overloaded):
            # Find the most overloaded facility
            overload_amounts = load - capacities
            overload_amounts[~overloaded] = -np.inf
            if not np.any(overload_amounts > 0):
                break
            f_over = int(np.argmax(overload_amounts))
            
            # Find customers in this facility
            custs_in_f = [c for c in range(n_cust) if result[c] == f_over]
            if not custs_in_f:
                break
                
            # Sort by demand descending (move big ones first to clear space)
            custs_in_f.sort(key=lambda c: demands[c], reverse=True)
            
            moved_any = False
            for c in custs_in_f:
                d_c = demands[c]
                # Find best destination
                best_f = -1
                best_cost = np.inf
                for f in range(n_fac):
                    if f == f_over:
                        continue
                    if load[f] + d_c <= capacities[f]:
                        # Calculate cost increase
                        # Change in assignment cost
                        delta_assign = assign_cost[c, f] - assign_cost[c, f_over]
                        # Change in open cost
                        delta_open = 0
                        if count[f] == 0:
                            delta_open = open_cost[f]
                        # Note: moving out of f_over might close it if count becomes 0, 
                        # but we are moving *from* overloaded, so it stays open or becomes less loaded.
                        # Actually, if count[f_over] becomes 0, we save open_cost[f_over].
                        # But since it was overloaded, it must have customers.
                        
                        total_delta = delta_assign + delta_open
                        if total_delta < best_cost:
                            best_cost = total_delta
                            best_f = f
                
                if best_f != -1:
                    # Perform move
                    result[c] = best_f
                    load[f_over] -= d_c
                    load[best_f] += d_c
                    count[f_over] -= 1
                    count[best_f] += 1
                    
                    # Update overloaded status
                    if load[f_over] <= capacities[f_over] + 1e-6:
                        overloaded[f_over] = False
                    moved_any = True
                    # If facility is still overloaded, continue to next customer in list
                    if not overloaded[f_over]:
                        break
                else:
                    # No feasible spot found? This shouldn't happen if total capacity >= total demand
                    # But just in case, pick any facility with space (even if it breaks optimality)
                    for f in range(n_fac):
                        if f != f_over and load[f] + d_c <= capacities[f]:
                            result[c] = f
                            load[f_over] -= d_c
                            load[best_f] += d_c
                            count[f_over] -= 1
                            count[best_f] += 1
                            moved_any = True
                            break

            if not moved_any:
                # Stuck, break to avoid infinite loop
                break
                
        return result, load

    def _subgradient_lagrangian():
        """
        Lagrangian Relaxation of capacity constraints.
        Minimize sum(assign_cost[i,j] * x_ij) + sum(open_cost[j] * y_j) + sum(lambda_j * (sum(demands[i]*x_ij) - capacity_j))
        Subject to sum_j x_ij = 1, x_ij <= y_j, x,y binary.
        
        For fixed lambda, the problem decomposes by customer i:
        Minimize assign_cost[i,j] + lambda_j * demands[i]
        Subject to x_ij <= y_j.
        
        This is equivalent to:
        For each customer i, choose j minimizing (assign_cost[i,j] + lambda_j * demands[i]).
        Then y_j is 1 if any customer is assigned to j.
        """
        # Initialize lambda to 0
        lam = np.zeros(n_fac)
        best_result = None
        best_cost = float('inf')
        
        # Parameters for subgradient
        step_size = 0.5
        max_iter = 500
        no_improve_count = 0
        
        for k in range(max_iter):
            if time.time() > deadline - 10.0:
                break
                
            # 1. Solve Lagrangian subproblem
            # Cost matrix for relaxed problem: assign_cost + outer(demands, lam)
            relaxed_costs = assign_cost + np.outer(demands, lam)
            
            # Assignment: argmin over j for each i
            assignment = np.argmin(relaxed_costs, axis=1)
            
            # Calculate loads and open facilities
            load = np.zeros(n_fac)
            for i in range(n_cust):
                load[assignment[i]] += demands[i]
            used = load > 0
            
            # Calculate Lagrangian Lower Bound (LB)
            # LB = sum(assign_cost[i, a_i]) + sum(open_cost[j] * y_j) + sum(lam_j * (load_j - cap_j))
            assign_sum = np.sum(assign_cost[np.arange(n_cust), assignment])
            open_sum = open_cost[used].sum()
            penalty_sum = np.sum(lam * (load - capacities))
            lb = assign_sum + open_sum + penalty_sum
            
            # Check feasibility
            if np.all(load <= capacities + 1e-6):
                # Feasible solution!
                current_cost = assign_sum + open_sum
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = assignment.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Infeasible, but we can repair it to get an upper bound
                repaired_result, repaired_load = _repair(assignment, load)
                # Verify repair
                if np.all(repaired_load <= capacities + 1e-6):
                    current_cost = _total_cost(repaired_result)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = repaired_result.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

            # 2. Update Subgradients
            # g_j = load_j - capacity_j
            g = load - capacities
            
            # Step size update (Polyak step size or constant)
            # We need an upper bound UB to calculate optimal step size
            # If we have a feasible solution, use it. Else, use a large number.
            if best_result is not None:
                ub = best_cost
            else:
                ub = lb + 1000.0 # Heuristic upper bound estimate
            
            gap = ub - lb
            if gap < 1e-4:
                break
                
            step = step_size * gap / (np.sum(g**2) + 1e-10)
            
            # Update lambda
            lam = lam + step * g
            lam = np.maximum(lam, 0) # Lambda must be non-negative
            
            # Adaptive step size decay
            if no_improve_count > 10:
                step_size *= 0.9
                if step_size < 0.01:
                    break

        return best_result, best_cost

    # Run Lagrangian Relaxation
    result, cost = _subgradient_lagrangian()
    
    if result is None:
        # Fallback to simple greedy if Lagrangian failed completely
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    break
        cost = _total_cost(result)

    # Local Search Polish
    def _local_search(result, load, deadline):
        result = result.copy()
        load = load.copy()
        count = np.zeros(n_fac, dtype=int)
        for c in range(n_cust):
            count[result[c]] += 1

        current_cost = _total_cost(result)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            best_delta = 0.0
            best_c = -1
            best_f = -1

            for c in range(n_cust):
                if time.time() > deadline - 3.0:
                    break
                old_f = result[c]
                d_c = demands[c]
                base_cost = assign_cost[c, old_f]

                for f in range(n_fac):
                    if f == old_f:
                        continue
                    if load[f] + d_c > capacities[f]:
                        continue

                    delta = assign_cost[c, f] - base_cost

                    if count[old_f] == 1:
                        delta -= open_cost[old_f]
                    if count[f] == 0:
                        delta += open_cost[f]

                    if delta < best_delta:
                        best_delta = delta
                        best_c = c
                        best_f = f

            if best_delta < -1e-9 and best_c != -1:
                c = best_c
                f = best_f
                old_f = result[c]
                result[c] = f
                load[old_f] -= demands[c]
                load[f] += demands[c]
                count[old_f] -= 1
                count[f] += 1
                current_cost += best_delta
                improved = True

        return result, current_cost

    # Get initial load for local search
    load = np.zeros(n_fac)
    for c in range(n_cust):
        load[result[c]] += demands[c]
        
    result, cost = _local_search(result, load, deadline)

    return result

# EVOLVE-BLOCK-END

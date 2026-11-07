"""Capacitated facility location: MILP (HiGHS) for exact solve, fallback to heuristic."""
import numpy as np
import time
import scipy.sparse as sp

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities.
    Primary: MILP via scipy.optimize.milp (HiGHS) for exact optimal solution.
    Fallback: Greedy + local search if MILP is unavailable or too slow.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # --- Try MILP first ---
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # First, run a fast greedy to get an initial upper bound (UB)
        def greedy_init_fast():
            result = np.zeros(n_cust, dtype=int)
            load = np.zeros(n_fac)
            order = np.argsort(-demands)
            for i in order:
                fac_order = np.argsort(assign_cost[i])
                assigned = False
                for f in fac_order:
                    if load[f] + demands[i] <= capacities[f] + 1e-6:
                        load[f] += demands[i]
                        result[i] = f
                        assigned = True
                        break
                if not assigned:
                    feasible_facs = np.where(capacities - load >= demands[i])[0]
                    if len(feasible_facs) > 0:
                        best_f = min(feasible_facs, key=lambda f: assign_cost[i, f])
                        load[best_f] += demands[i]
                        result[i] = best_f
                    else:
                        return None, None
            # Compute cost
            fac_count = np.zeros(n_fac, dtype=int)
            for f in result:
                fac_count[f] += 1
            cost = np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])
            return result, cost

        init_res, init_cost = greedy_init_fast()
        initial_ub = init_cost if init_cost is not None else None

        # Variables: x[i, j] for i in 0..n_cust-1, j in 0..n_fac-1
        # y[j] for j in 0..n_fac-1
        # Total variables: n_cust * n_fac + n_fac
        
        # Objective coefficients
        c_obj = np.zeros(n_cust * n_fac + n_fac)
        c_obj[:n_cust * n_fac] = assign_cost.flatten()
        c_obj[n_cust * n_fac:] = open_cost
        
        # Bounds
        lb = np.zeros(n_cust * n_fac + n_fac)
        ub = np.ones(n_cust * n_fac + n_fac)
        
        # Integrality
        integrality = np.ones(n_cust * n_fac + n_fac)
        
        # Constraints
        # 1. Assignment: sum_j x[i, j] = 1 for each i
        # Use sparse matrix for efficiency
        A_eq_rows = np.arange(n_cust)
        A_eq_cols = np.repeat(np.arange(n_fac), n_cust) + np.tile(np.arange(n_cust) * n_fac, n_fac)
        # Wait, the indexing is: x[i, j] is at index i * n_fac + j
        # So for row i, cols are i*n_fac + 0, i*n_fac + 1, ..., i*n_fac + n_fac-1
        A_eq_rows = np.repeat(np.arange(n_cust), n_fac)
        A_eq_cols = np.array([i * n_fac + j for i in range(n_cust) for j in range(n_fac)])
        A_eq_data = np.ones(n_cust * n_fac)
        A_eq = sp.csr_matrix((A_eq_data, (A_eq_rows, A_eq_cols)), shape=(n_cust, n_cust * n_fac + n_fac))
        b_eq = np.ones(n_cust)
        
        # 2. Capacity: sum_i d_i * x[i, j] - C_j * y[j] <= 0 for each j
        A_ub_rows = np.repeat(np.arange(n_fac), n_cust)
        A_ub_cols_x = np.array([i * n_fac + j for i in range(n_cust) for j in range(n_fac)])
        # For each j, we have n_cust entries for x[i, j] and 1 entry for y[j]
        # Total rows = n_fac, but each row has n_cust + 1 non-zeros
        # Let's build it properly
        A_ub_rows_list = []
        A_ub_cols_list = []
        A_ub_data_list = []
        for j in range(n_fac):
            for i in range(n_cust):
                A_ub_rows_list.append(j)
                A_ub_cols_list.append(i * n_fac + j)
                A_ub_data_list.append(demands[i])
            A_ub_rows_list.append(j)
            A_ub_cols_list.append(n_cust * n_fac + j)
            A_ub_data_list.append(-capacities[j])
        
        A_ub = sp.csr_matrix((A_ub_data_list, (A_ub_rows_list, A_ub_cols_list)), shape=(n_fac, n_cust * n_fac + n_fac))
        b_ub = np.zeros(n_fac)
        
        constraints = [
            LinearConstraint(A_eq, b_eq, b_eq),
            LinearConstraint(A_ub, -np.inf, b_ub)
        ]
        
        bounds = Bounds(lb, ub)
        
        # Solve with time limit and initial bound
        options = {"time_limit": 130, "mip_rel_gap": 0.0}
        if initial_ub is not None:
            options["mip_max_nodes"] = 1000000  # Allow many nodes
            # HiGHS doesn't directly accept an initial UB in the same way, 
            # but having a good solution helps. We can pass the solution via a different mechanism?
            # Actually, scipy.optimize.milp doesn't have a direct "initial_solution" parameter in older versions.
            # However, HiGHS might benefit from the time limit being set appropriately.
            # Let's just rely on the solver's internal heuristics.
            
        result = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds, options=options)
        
        if result.success or (hasattr(result, 'status') and result.status == 0):
            x_flat = result.x[:n_cust * n_fac].reshape(n_cust, n_fac)
            # Find argmax for each customer (should be 1.0 at best location)
            # Round to handle numerical noise
            x_flat = np.where(x_flat > 0.5, 1.0, 0.0)
            
            # Check if any customer is unassigned (sum should be 1)
            row_sums = x_flat.sum(axis=1)
            if np.allclose(row_sums, 1.0):
                assignment = np.argmax(x_flat, axis=1)
                
                # Verify feasibility
                load = np.zeros(n_fac)
                for i, f in enumerate(assignment):
                    load[f] += demands[i]
                if np.all(load <= capacities + 1e-6):
                    return assignment.astype(int)
                else:
                    # If infeasible due to numerical issues, fall back
                    pass
        else:
            # Fallback
            pass
            
    except Exception:
        # Fallback
        pass

    # --- Fallback: Greedy + Local Search ---
    
    def compute_true_cost(result):
        """Compute true cost for a feasible assignment."""
        if result is None:
            return float('inf')
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
            fac_count[f] += 1
        if np.any(load > capacities + 1e-6):
            return float('inf')
        return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])

    def repair_assignment(result, load):
        """
        Repair an infeasible assignment by moving customers from overloaded facilities.
        Returns a feasible assignment and its load.
        """
        result = result.astype(int).copy()
        load = load.copy()
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            fac_count[f] += 1
            
        # Identify overloaded facilities
        overloaded = np.where(load > capacities + 1e-6)[0]
        
        # If no overloaded, it's feasible
        if len(overloaded) == 0:
            return result, load, fac_count

        # We need to move customers out of overloaded facilities
        # Sort customers in overloaded facilities by "cost of moving" to next best facility
        # We'll iterate until feasible or no moves possible
        
        max_iterations = 100
        for _ in range(max_iterations):
            overloaded = np.where(load > capacities + 1e-6)[0]
            if len(overloaded) == 0:
                break
            
            # Find a customer to move from an overloaded facility
            # Prioritize customers with the smallest penalty for moving
            # Penalty = min_cost_to_other_facility - current_cost
            
            best_move_customer = -1
            best_move_from = -1
            best_move_to = -1
            min_penalty = float('inf')
            
            for f_from in overloaded:
                # Get customers assigned to f_from
                customers_in_f = np.where(result == f_from)[0]
                for i in customers_in_f:
                    # Find best alternative facility
                    # Must have capacity
                    # Cost = assign_cost[i, f_to] - assign_cost[i, f_from]
                    # We want to minimize the increase in cost
                    # Also, we need to ensure f_to has space
                    feasible_to = []
                    for f_to in range(n_fac):
                        if f_to == f_from:
                            continue
                        if load[f_to] + demands[i] <= capacities[f_to] + 1e-6:
                            # Calculate penalty
                            # If f_to is currently closed, we pay open_cost
                            # If f_from becomes empty, we save open_cost
                            # For simplicity in repair, we just look at assignment cost difference
                            # and assume open costs are handled by the final cost calculation
                            # But to be safe, let's just pick the one with lowest assignment cost increase
                            penalty = assign_cost[i, f_to] - assign_cost[i, f_from]
                            feasible_to.append((penalty, f_to))
                    
                    if feasible_to:
                        best_penalty, best_f_to = min(feasible_to, key=lambda x: x[0])
                        if best_penalty < min_penalty:
                            min_penalty = best_penalty
                            best_move_customer = i
                            best_move_from = f_from
                            best_move_to = best_f_to
            
            if best_move_customer == -1:
                # No feasible move found? This should not happen if total capacity > total demand
                # If it does, we have a problem. Return infeasible or best effort.
                break
            
            # Perform the move
            i = best_move_customer
            f_from = best_move_from
            f_to = best_move_to
            
            result[i] = f_to
            load[f_from] -= demands[i]
            load[f_to] += demands[i]
            fac_count[f_from] -= 1
            fac_count[f_to] += 1
            
        return result, load, fac_count

    def local_search(result, time_limit):
        """Local search with 1-moves and 2-swaps."""
        current = result.astype(int).copy()
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(current):
            load[f] += demands[i]
            fac_count[f] += 1
            
        def get_cost():
            return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), current])
            
        current_cost = get_cost()
        best_cost = current_cost
        best_result = current.copy()
        
        start = time.time()
        improved = True
        
        while improved and (time.time() - start < time_limit):
            improved = False
            
            # 1. Single Reassignment (1-move)
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                cur_f = current[i]
                d_i = demands[i]
                best_delta = 0
                best_new_f = -1
                
                for new_f in range(n_fac):
                    if new_f == cur_f:
                        continue
                    if load[new_f] + d_i > capacities[new_f] + 1e-6:
                        continue
                    
                    delta = assign_cost[i, new_f] - assign_cost[i, cur_f]
                    
                    if fac_count[cur_f] == 1:
                        delta -= open_cost[cur_f]
                    if fac_count[new_f] == 0:
                        delta += open_cost[new_f]
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_new_f = new_f
                        
                if best_new_f != -1:
                    old_f = cur_f
                    new_f = best_new_f
                    current[i] = new_f
                    load[old_f] -= d_i
                    load[new_f] += d_i
                    fac_count[old_f] -= 1
                    fac_count[new_f] += 1
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()
            
            # 2. Swap (2-move)
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                f_i = current[i]
                d_i = demands[i]
                for j in range(i + 1, n_cust):
                    f_j = current[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    if load[f_i] - d_i + d_j > capacities[f_i] + 1e-6:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j] + 1e-6:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    # Check open cost changes for swap
                    # If f_i becomes empty, save open_cost[f_i]
                    # If f_j becomes empty, save open_cost[f_j]
                    # If f_i becomes non-empty (was empty), add open_cost[f_i]
                    # If f_j becomes non-empty (was empty), add open_cost[f_j]
                    # Since we are swapping, both facilities remain non-empty if they were before.
                    # But if one was count 1, it becomes count 1 (swapped), so no change in count?
                    # Wait, if customer i leaves f_i, and customer j leaves f_j.
                    # i goes to f_j, j goes to f_i.
                    # Count of f_i: -1 (i leaves) + 1 (j enters) = 0 change.
                    # Count of f_j: -1 (j leaves) + 1 (i enters) = 0 change.
                    # So open costs do not change in a swap between two non-empty facilities.
                    
                    if delta < 0:
                        current[i] = f_j
                        current[j] = f_i
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = current.copy()
                        break
            
        return best_result

    def greedy_init():
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-6:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # Fallback: find any feasible
                feasible_facs = np.where(capacities - load >= demands[i])[0]
                if len(feasible_facs) > 0:
                    best_f = min(feasible_facs, key=lambda f: assign_cost[i, f])
                    load[best_f] += demands[i]
                    result[i] = best_f
                else:
                    return None
        return result

    best_feasible_result = None
    best_feasible_cost = float('inf')
    
    init_res = greedy_init()
    if init_res is not None:
        init_cost = compute_true_cost(init_res)
        if init_cost < best_feasible_cost:
            best_feasible_cost = init_cost
            best_feasible_result = init_res.copy()
            
    if best_feasible_result is not None:
        time_elapsed = time.time() - start_time
        time_remaining = deadline - time_elapsed - 5.0 # Safety margin
        
        if time_remaining > 1.0:
            polished = local_search(best_feasible_result, time_remaining)
            if polished is not None:
                polished_cost = compute_true_cost(polished)
                if polished_cost < best_feasible_cost:
                    best_feasible_result = polished
    
    if best_feasible_result is None:
        # Fallback to simple greedy if something went wrong
        best_feasible_result = greedy_init()
        if best_feasible_result is None:
             # Last resort: assign to facility 0 (might be infeasible, but better than crash)
             best_feasible_result = np.zeros(n_cust, dtype=int)
             
    return best_feasible_result

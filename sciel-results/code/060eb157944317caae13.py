"""Capacitated facility location: MILP (HiGHS) with LP-relaxation seeding, Lagrangian relaxation, and heuristic polish."""
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities.
    Primary: MILP via scipy.optimize.milp (HiGHS).
    Fallback: LP relaxation + rounding + local search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def compute_cost(result):
        """Compute total cost for a given assignment."""
        if result is None:
            return float('inf')
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
            fac_count[f] += 1
        if np.any(load > capacities + 1e-9):
            return float('inf')
        return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])

    def greedy_init(order_type='desc', randomize=False, seed=None):
        """Greedy initialization."""
        if seed is not None:
            np.random.seed(seed)
            
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        if order_type == 'desc':
            order = np.argsort(-demands)
        else:
            order = np.argsort(demands)
            
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            if randomize:
                noise = np.random.uniform(0, 10) * np.ones(n_fac)
                fac_order = np.argsort(assign_cost[i] + noise)
                
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            
            if not assigned:
                remaining = capacities - load
                feasible_facs = np.where(remaining >= demands[i])[0]
                if len(feasible_facs) > 0:
                    best_f = min(feasible_facs, key=lambda f: assign_cost[i, f])
                    load[best_f] += demands[i]
                    result[i] = best_f
                else:
                    return None
                    
        return result

    def lp_rounding_init():
        """LP relaxation + randomized rounding."""
        try:
            from scipy.optimize import linprog
            # Variables: x[i,j] (n_cust*n_fac) + y[j] (n_fac)
            n_vars = n_cust * n_fac + n_fac
            c_obj = np.concatenate([assign_cost.flatten(), open_cost])
            bounds = [(0, 1)] * n_vars
            
            # Equality: sum_j x[i,j] = 1
            A_eq = np.zeros((n_cust, n_vars))
            for i in range(n_cust):
                for j in range(n_fac):
                    A_eq[i, i * n_fac + j] = 1.0
            b_eq = np.ones(n_cust)
            
            # Inequality: capacity constraints
            # sum_i demand[i]*x[i,j] - capacity[j]*y[j] <= 0
            A_ub = np.zeros((n_fac, n_vars))
            b_ub = np.zeros(n_fac)
            for j in range(n_fac):
                for i in range(n_cust):
                    A_ub[j, i * n_fac + j] = demands[i]
                A_ub[j, n_cust * n_fac + j] = -capacities[j]
            
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.status == 0 and res.x is not None:
                x_lp = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
                # Round: assign to argmax x_lp[i]
                result = np.argmax(x_lp, axis=1).astype(int)
                return result
        except Exception:
            pass
        return None

    def repair_assignment(result):
        """Repair capacity violations by reassigning customers to least-loaded feasible facilities."""
        result = result.astype(int).copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        
        # Find facilities over capacity
        violated = load > capacities + 1e-9
        if not np.any(violated):
            return result
        
        # Reassign customers from over-capacity facilities
        for f in np.where(violated)[0]:
            # Get customers assigned to this facility
            custs = np.where(result == f)[0]
            # Sort by demand descending (move largest first)
            custs = custs[np.argsort(-demands[custs])]
            
            for i in custs:
                d_i = demands[i]
                # Find best feasible facility
                feasible = []
                for nf in range(n_fac):
                    if load[nf] + d_i <= capacities[nf]:
                        feasible.append((assign_cost[i, nf], nf))
                
                if feasible:
                    feasible.sort()
                    best_nf = feasible[0][1]
                    load[f] -= d_i
                    load[best_nf] += d_i
                    result[i] = best_nf
                else:
                    # No feasible facility; try to swap with another customer
                    for j in range(n_cust):
                        if j == i or result[j] == f:
                            continue
                        d_j = demands[j]
                        cur_f_j = result[j]
                        if load[cur_f_j] - d_j + d_i <= capacities[cur_f_j] and load[f] - d_i + d_j <= capacities[f]:
                            # Swap i and j
                            load[f] -= d_i
                            load[f] += d_j
                            load[cur_f_j] -= d_j
                            load[cur_f_j] += d_i
                            result[i] = cur_f_j
                            result[j] = f
                            break
        
        # Final check
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        if np.any(load > capacities + 1e-9):
            return None
        return result

    def lagrangian_init(time_limit=10.0):
        """Lagrangian relaxation with subgradient method."""
        start = time.time()
        
        # Multipliers for capacity constraints
        lambda_ = np.zeros(n_fac)
        best_result = None
        best_dual_bound = float('inf')
        best_primal = float('inf')
        
        # Step size parameters
        best_known = float('inf')
        alpha = 1.0
        step_count = 0
        
        while time.time() - start < time_limit:
            # Solve relaxed problem: for each customer, assign to argmin_j (assign_cost[i,j] + lambda_[j] * demand[i])
            # But we also need to account for open costs
            # Relaxed objective: sum_i,j x[i,j]*(assign_cost[i,j] + lambda_[j]*demand[i]) + sum_j open_cost[j]*y[j]
            # With x[i,j] <= y[j], the relaxed problem decomposes per customer
            # For each customer i: min_j (assign_cost[i,j] + lambda_[j]*demand[i]) + (open_cost[j] if y[j]=1)
            # This is still coupled through y[j], so we use a heuristic approach
            
            # Simpler approach: assign each customer to argmin_j (assign_cost[i,j] + lambda_[j]*demand[i])
            # Then repair capacity violations
            relaxed_cost = assign_cost + np.outer(demands, lambda_)
            result = np.argmin(relaxed_cost, axis=1).astype(int)
            
            # Compute primal cost (with open costs)
            load = np.zeros(n_fac)
            fac_count = np.zeros(n_fac, dtype=int)
            for i in range(n_cust):
                load[result[i]] += demands[i]
                fac_count[result[i]] += 1
            
            primal_cost = np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])
            
            # Check capacity feasibility
            if np.all(load <= capacities + 1e-9):
                if primal_cost < best_primal:
                    best_primal = primal_cost
                    best_result = result.copy()
            else:
                # Try to repair
                repaired = repair_assignment(result)
                if repaired is not None:
                    if primal_cost < best_primal:
                        best_primal = primal_cost
                        best_result = repaired.copy()
            
            # Dual bound: sum_i min_j (assign_cost[i,j] + lambda_[j]*demand[i]) + max(0, sum_j lambda_[j]*capacities[j] - sum_j lambda_[j]*capacities[j])
            # Actually: dual bound = sum_i min_j (assign_cost[i,j] + lambda_[j]*demand[i]) + sum_j lambda_[j]*capacities[j]
            # (since the Lagrangian dual is max_lambda L(lambda) where L includes the capacity terms)
            min_costs = np.min(relaxed_cost, axis=1)
            dual_bound = np.sum(min_costs) + np.sum(lambda_ * capacities)
            
            if dual_bound < best_dual_bound:
                best_dual_bound = dual_bound
            
            # Update best known upper bound
            if primal_cost < best_known:
                best_known = primal_cost
            
            # Subgradient: g_j = demand_assigned_to_j - capacity[j]
            g = load - capacities
            
            # Step size: alpha * (best_known - dual_bound) / ||g||^2
            g_norm_sq = np.sum(g ** 2)
            if g_norm_sq > 1e-12:
                step = alpha * (best_known - dual_bound) / g_norm_sq
            else:
                step = 0.0
            
            # Update multipliers (non-negative)
            lambda_ = np.maximum(0, lambda_ + step * g)
            
            # Decay alpha
            step_count += 1
            if step_count % 10 == 0:
                alpha = max(0.1, alpha * 0.95)
            
            # Check convergence
            if abs(best_known - dual_bound) < 1e-4:
                break
        
        if best_result is not None:
            return best_result
        return None

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
                    if load[new_f] + d_i > capacities[new_f]:
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
                    
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
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

    # --- Try MILP first ---
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Variables: x[i, j] for i in range(n_cust), j in range(n_fac), plus y[j] for j in range(n_fac)
        # Total variables: n_cust * n_fac + n_fac
        n_vars = n_cust * n_fac + n_fac
        
        # Objective coefficients
        c = np.zeros(n_vars)
        for i in range(n_cust):
            for j in range(n_fac):
                c[i * n_fac + j] = assign_cost[i, j]
        for j in range(n_fac):
            c[n_cust * n_fac + j] = open_cost[j]
        
        # Integrality: x variables are binary (0/1), y variables are binary (0/1)
        integrality = np.ones(n_vars)
        
        # Bounds: all variables in [0, 1]
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        
        # Constraints:
        # 1. Each customer assigned to exactly one facility: sum_j x[i, j] = 1 for all i
        #    n_cust equality constraints
        # 2. Capacity: sum_i demand[i] * x[i, j] <= capacity[j] * y[j] for all j
        #    => sum_i demand[i] * x[i, j] - capacity[j] * y[j] <= 0
        # 3. Linking: x[i, j] <= y[j] for all i, j
        #    => x[i, j] - y[j] <= 0
        
        n_constraints = n_cust + n_fac + n_cust * n_fac
        
        # Build constraint matrix
        A = np.zeros((n_constraints, n_vars))
        lb_cons = np.zeros(n_constraints)
        ub_cons = np.zeros(n_constraints)
        
        # Equality constraints: sum_j x[i, j] = 1
        for i in range(n_cust):
            for j in range(n_fac):
                A[i, i * n_fac + j] = 1.0
            lb_cons[i] = 1.0
            ub_cons[i] = 1.0
        
        # Capacity constraints: sum_i demand[i] * x[i, j] - capacity[j] * y[j] <= 0
        row_offset = n_cust
        for j in range(n_fac):
            for i in range(n_cust):
                A[row_offset + j, i * n_fac + j] = demands[i]
            A[row_offset + j, n_cust * n_fac + j] = -capacities[j]
            lb_cons[row_offset + j] = -np.inf
            ub_cons[row_offset + j] = 0.0
        
        # Linking constraints: x[i, j] - y[j] <= 0
        row_offset = n_cust + n_fac
        for i in range(n_cust):
            for j in range(n_fac):
                A[row_offset + i * n_fac + j, i * n_fac + j] = 1.0
                A[row_offset + i * n_fac + j, n_cust * n_fac + j] = -1.0
                lb_cons[row_offset + i * n_fac + j] = -np.inf
                ub_cons[row_offset + i * n_fac + j] = 0.0
        
        constraint = LinearConstraint(A, lb_cons, ub_cons)
        bounds = Bounds(lb, ub)
        
        res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds,
                   options={"time_limit": 130, "mip_rel_gap": 0.0})
        
        if res.status == 0 and res.x is not None:
            x_sol = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
            # Round to nearest integer (should be very close to 0 or 1)
            result = np.argmax(x_sol, axis=1).astype(int)
            
            # Verify feasibility
            load = np.zeros(n_fac)
            for i in range(n_cust):
                load[result[i]] += demands[i]
            if np.all(load <= capacities + 1e-6):
                return result
        
        # If MILP didn't find a solution or infeasible, fall through to heuristic
        
    except Exception:
        pass
    
    # --- Fallback: Greedy + LP + Lagrangian + Local Search ---
    
    candidates = []
    
    # LP-based initialization
    lp_init = lp_rounding_init()
    if lp_init is not None:
        candidates.append(lp_init)
    
    # Lagrangian relaxation initialization
    lag_time = min(8.0, max(0, deadline - (time.time() - start_time) - 15.0))
    if lag_time > 1.0:
        lag_init = lagrangian_init(time_limit=lag_time)
        if lag_init is not None:
            candidates.append(lag_init)
    
    # Greedy initializations
    init_1 = greedy_init('desc')
    if init_1 is not None:
        candidates.append(init_1)
        
    init_2 = greedy_init('asc')
    if init_2 is not None:
        candidates.append(init_2)
        
    for s in [42, 123, 456]:
        if time.time() > deadline - 5:
            break
        init_r = greedy_init('desc', randomize=True, seed=s)
        if init_r is not None:
            candidates.append(init_r)
    
    if not candidates:
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        candidates.append(result)
        
    best_init = min(candidates, key=compute_cost)
    
    time_elapsed = time.time() - start_time
    time_remaining = deadline - time_elapsed
    
    if time_remaining > 2.0:
        polished = local_search(best_init, time_remaining - 3.0)
        if polished is not None:
            best_init = polished
    
    return best_init

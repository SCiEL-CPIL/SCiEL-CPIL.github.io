# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Uses MILP via scipy.optimize.milp (HiGHS) for exact/near-exact solution,
    with greedy fallback and enhanced local search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin
    
    def _compute_obj(result):
        """Compute objective for a valid assignment."""
        if result is None:
            return float('inf')
        load = np.zeros(n_fac, dtype=np.int64)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
        if np.any(load > capacities):
            return float('inf')
        open_mask = np.zeros(n_fac, dtype=bool)
        for f in result:
            open_mask[f] = True
        return np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(n_cust), result])
    
    # Greedy fallback
    def _greedy(order):
        fac_order = np.argsort(assign_cost, axis=1)
        load = np.zeros(n_fac, dtype=np.int64)
        result = np.full(n_cust, -1, dtype=np.int64)
        for i in order:
            assigned = False
            for f in fac_order[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                return None
        return result
    
    best_result = None
    best_cost = float('inf')
    
    # Try greedy with largest demand first
    order1 = np.argsort(-demands)
    r = _greedy(order1)
    if r is not None:
        c = _compute_obj(r)
        if c < best_cost:
            best_cost = c
            best_result = r.copy()
    
    # Try MILP approach
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Variables: x[i*n_fac + j] = 1 if customer i assigned to facility j
        #             x[n_cust*n_fac + j] = 1 if facility j is open
        n_vars = n_cust * n_fac + n_fac
        
        # Objective: minimize sum(open_cost[j] * y[j]) + sum(assign_cost[i,j] * x[i,j])
        c_obj = np.zeros(n_vars, dtype=np.float64)
        for i in range(n_cust):
            for j in range(n_fac):
                c_obj[i * n_fac + j] = assign_cost[i, j]
        for j in range(n_fac):
            c_obj[n_cust * n_fac + j] = open_cost[j]
        
        # Constraints:
        # 1. For each customer i: sum_j x[i,j] = 1
        # 2. For each facility j: sum_i demands[i]*x[i,j] <= capacities[j]*y[j]
        # 3. For each i,j: x[i,j] <= y[j]
        
        n_constraints = n_cust + n_fac + n_cust * n_fac
        
        A = np.zeros((n_constraints, n_vars), dtype=np.float64)
        lb = np.zeros(n_constraints, dtype=np.float64)
        ub = np.zeros(n_constraints, dtype=np.float64)
        
        # Constraint 1: sum_j x[i,j] = 1
        for i in range(n_cust):
            for j in range(n_fac):
                A[i, i * n_fac + j] = 1.0
            lb[i] = 1.0
            ub[i] = 1.0
        
        # Constraint 2: sum_i demands[i]*x[i,j] <= capacities[j]*y[j]
        # => sum_i demands[i]*x[i,j] - capacities[j]*y[j] <= 0
        for j in range(n_fac):
            row = n_cust + j
            for i in range(n_cust):
                A[row, i * n_fac + j] = demands[i]
            A[row, n_cust * n_fac + j] = -capacities[j]
            lb[row] = -np.inf
            ub[row] = 0.0
        
        # Constraint 3: x[i,j] - y[j] <= 0
        for i in range(n_cust):
            for j in range(n_fac):
                row = n_cust + n_fac + i * n_fac + j
                A[row, i * n_fac + j] = 1.0
                A[row, n_cust * n_fac + j] = -1.0
                lb[row] = -np.inf
                ub[row] = 0.0
        
        # Variable bounds: 0 <= x <= 1, 0 <= y <= 1
        var_lb = np.zeros(n_vars, dtype=np.float64)
        var_ub = np.ones(n_vars, dtype=np.float64)
        
        # Integrality: x and y are binary (0 or 1)
        integrality = np.ones(n_vars, dtype=np.int8)
        
        constraint = LinearConstraint(A, lb, ub)
        bounds = Bounds(var_lb, var_ub)
        
        start_time = time.time()
        res = milp(
            c=c_obj,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 100.0, "mip_rel_gap": 0.01}
        )
        
        if res.status == 0 or res.status == 2:  # Optimal or best found
            x_sol = res.x
            if x_sol is not None:
                result_milp = np.zeros(n_cust, dtype=np.int64)
                valid = True
                for i in range(n_cust):
                    # Find which facility
                    assigned = False
                    for j in range(n_fac):
                        if x_sol[i * n_fac + j] > 0.5:
                            result_milp[i] = j
                            assigned = True
                            break
                    if not assigned:
                        valid = False
                        break
                
                if valid:
                    c_milp = _compute_obj(result_milp)
                    if c_milp < best_cost:
                        best_cost = c_milp
                        best_result = result_milp.copy()
        
    except Exception as e:
        pass
    
    # If MILP didn't give a good result, try local search on best greedy
    if best_result is not None and time.time() < deadline - 5:
        fac_order = np.argsort(assign_cost, axis=1)
        
        def _local_search(result, max_iters=500):
            result = result.copy()
            load = np.zeros(n_fac, dtype=np.int64)
            for i, f in enumerate(result):
                load[f] += demands[i]
            
            current_obj = _compute_obj(result)
            
            improved = True
            iters = 0
            while improved and iters < max_iters and time.time() < deadline:
                improved = False
                iters += 1
                for i in range(n_cust):
                    if time.time() > deadline:
                        break
                    cur_f = result[i]
                    for f in fac_order[i]:
                        if f == cur_f:
                            continue
                        if load[f] + demands[i] > capacities[f]:
                            continue
                        delta_assign = assign_cost[i, f] - assign_cost[i, cur_f]
                        delta_open = 0.0
                        if load[cur_f] - demands[i] == 0:
                            delta_open -= open_cost[cur_f]
                        if load[f] == 0:
                            delta_open += open_cost[f]
                        total_delta = delta_assign + delta_open
                        if total_delta < -1e-9:
                            load[cur_f] -= demands[i]
                            load[f] += demands[i]
                            result[i] = f
                            current_obj += total_delta
                            improved = True
                            break
            
            # 2-opt swaps
            improved = True
            swap_iters = 0
            while improved and swap_iters < 200 and time.time() < deadline:
                improved = False
                swap_iters += 1
                for i in range(n_cust):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, n_cust):
                        if time.time() > deadline:
                            break
                        fi = result[i]
                        fj = result[j]
                        if fi == fj:
                            continue
                        # Try swapping i->fj and j->fi
                        if load[fj] - demands[j] + demands[i] <= capacities[fj] and \
                           load[fi] - demands[i] + demands[j] <= capacities[fi]:
                            delta = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                                    (assign_cost[i, fi] + assign_cost[j, fj])
                            # Check open cost changes
                            delta_open = 0.0
                            new_load_i = load[fi] - demands[i]
                            new_load_j = load[fj] - demands[j]
                            if new_load_i == 0:
                                delta_open -= open_cost[fi]
                            if new_load_j == 0:
                                delta_open -= open_cost[fj]
                            if load[fi] - demands[i] + demands[j] == 0:
                                pass  # fi already closed or stays open
                            if load[fj] - demands[j] + demands[i] == 0:
                                pass  # fj already closed or stays open
                            # More careful: check if fi becomes empty
                            if load[fi] - demands[i] == 0:
                                delta_open -= open_cost[fi]
                            if load[fj] - demands[j] == 0:
                                delta_open -= open_cost[fj]
                            total_delta = delta + delta_open
                            if total_delta < -1e-9:
                                load[fi] = load[fi] - demands[i] + demands[j]
                                load[fj] = load[fj] - demands[j] + demands[i]
                                result[i] = fj
                                result[j] = fi
                                current_obj += total_delta
                                improved = True
                                break
                    if improved:
                        break
            
            return result
        
        best_result = _local_search(best_result, max_iters=500)
        best_cost = _compute_obj(best_result)
    
    if best_result is None:
        # Ultimate fallback
        result = np.array([i % n_fac for i in range(n_cust)], dtype=np.int64)
        return result
    
    return best_result.astype(int)

# EVOLVE-BLOCK-END

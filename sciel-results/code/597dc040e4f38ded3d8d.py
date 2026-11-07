import numpy as np
import time
import scipy.optimize as opt
import scipy.sparse as sp

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # Try MILP first
    try:
        # Variable ordering: x_00...x_0,19, x_10...x_99,19, y_0...y_19
        n_vars = n_cust * n_fac + n_fac
        
        # Cost vector
        c = np.zeros(n_vars)
        # Assignment costs
        c[:n_cust * n_fac] = assign_cost.flatten()
        # Open costs
        c[n_cust * n_fac:] = open_cost
        
        # Constraints
        # 1. Each customer assigned to exactly one facility
        # Sum_j x_ij = 1
        A_assign = sp.identity(n_cust, n_vars, format='coo') 
        # This is wrong. A_assign needs to pick the right columns.
        # Let's build it manually or use sparse construction.
        
        # Better approach: build sparse matrices directly
        rows_assign = []
        cols_assign = []
        data_assign = []
        for i in range(n_cust):
            for j in range(n_fac):
                rows_assign.append(i)
                cols_assign.append(i * n_fac + j)
                data_assign.append(1.0)
        A_assign = sp.coo_matrix((data_assign, (rows_assign, cols_assign)), shape=(n_cust, n_vars)).tocsr()
        b_assign = np.ones(n_cust)
        
        # 2. Capacity constraints
        # Sum_i demands[i] * x_ij <= capacities[j]
        rows_cap = []
        cols_cap = []
        data_cap = []
        for j in range(n_fac):
            for i in range(n_cust):
                rows_cap.append(j)
                cols_cap.append(i * n_fac + j)
                data_cap.append(demands[i])
        A_cap = sp.coo_matrix((data_cap, (rows_cap, cols_cap)), shape=(n_fac, n_vars)).tocsr()
        b_cap = capacities
        
        # 3. Linking constraints
        # x_ij - y_j <= 0
        rows_link = []
        cols_link = []
        data_link = []
        for i in range(n_cust):
            for j in range(n_fac):
                rows_link.append(i * n_fac + j) # Just a unique ID for the row
                cols_link.append(i * n_fac + j)
                cols_link.append(n_cust * n_fac + j)
                data_link.append(1.0)
                data_link.append(-1.0)
        A_link = sp.coo_matrix((data_link, (rows_link, cols_link)), shape=(n_cust * n_fac, n_vars)).tocsr()
        b_link = np.zeros(n_cust * n_fac)
        
        # Combine constraints
        A = sp.vstack([A_assign, A_cap, A_link]).tocsr()
        b = np.concatenate([b_assign, b_cap, b_link])
        
        # Bounds
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        
        # Integrality
        integrality = np.ones(n_vars) # All variables are integer
        
        # Solve
        # Options for HiGHS
        options = {
            "time_limit": 100,
            "presolve": "on"
        }
        
        res = opt.milp(c, constraints=A, integrality=integrality, bounds=ub, options=options)
        
        # Note: scipy.milp takes bounds as (lb, ub) tuple or array of tuples?
        # Documentation: bounds : sequence of (min, max) pairs or array_like
        # If I pass `ub`, it assumes lb=0? No, I should pass properly.
        # Let's pass bounds as a list of tuples or array of shape (n_vars, 2)
        bounds_arr = np.zeros((n_vars, 2))
        bounds_arr[:, 0] = 0
        bounds_arr[:, 1] = 1
        
        res = opt.milp(c, constraints=A, integrality=integrality, bounds=bounds_arr, options=options)
        
        if res.status == 0: # Optimal
            x_sol = res.x
            assignment = np.zeros(n_cust, dtype=int)
            for i in range(n_cust):
                # Find j where x_sol[i*n_fac + j] > 0.5
                xs = x_sol[i*n_fac : (i+1)*n_fac]
                j = np.argmax(xs)
                if xs[j] > 0.5:
                    assignment[i] = j
                else:
                    # Should not happen if optimal
                    assignment[i] = j 
                    
            # Verify
            load = np.zeros(n_fac)
            for i in range(n_cust):
                load[assignment[i]] += demands[i]
            
            if np.all(load <= capacities + 1e-6):
                return assignment
            else:
                # Fallback to heuristic if MILP result is somehow infeasible (shouldn't happen)
                pass
                
    except Exception as e:
        pass
        
    # Fallback Heuristic
    # Greedy + Local Search
    
    def compute_cost(result, is_open):
        if not np.all(is_open[result]):
            return float('inf')
        load = np.zeros(n_fac)
        for i, f in enumerate(result):
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return float('inf')
        open_sum = np.sum(open_cost[is_open])
        assign_sum = np.sum(assign_cost[np.arange(n_cust), result])
        return open_sum + assign_sum

    def greedy_init():
        result = np.zeros(n_cust, dtype=int)
        is_open = np.ones(n_fac, dtype=bool)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            costs = assign_cost[i].copy()
            fac_order = np.argsort(costs)
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                remaining = capacities - load
                feasible = np.where(remaining >= demands[i])[0]
                if len(feasible) == 0:
                    return None, None
                f = feasible[np.argmax(remaining[feasible])]
                load[f] += demands[i]
                result[i] = f
        return result, is_open

    def local_search(result, is_open):
        current_cost = compute_cost(result, is_open)
        if np.isinf(current_cost):
            return result, is_open, current_cost

        load = np.zeros(n_fac)
        for i, f in enumerate(result):
            load[f] += demands[i]
        fac_counts = np.zeros(n_fac, dtype=int)
        for f in result:
            fac_counts[f] += 1
            
        improved = True
        passes = 0
        max_passes = 100
        
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            
            best_move_delta = 0
            best_move_i = -1
            best_move_f_new = -1
            
            for i in range(n_cust):
                f_old = result[i]
                for f_new in range(n_fac):
                    if f_new == f_old:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new]:
                        continue
                    delta = assign_cost[i, f_new] - assign_cost[i, f_old]
                    if not is_open[f_new]:
                        delta += open_cost[f_new]
                    if fac_counts[f_old] == 1:
                        delta -= open_cost[f_old]
                    if delta < best_move_delta:
                        best_move_delta = delta
                        best_move_i = i
                        best_move_f_new = f_new
                        
            if best_move_i != -1:
                i = best_move_i
                f_old = result[i]
                f_new = best_move_f_new
                result[i] = f_new
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                fac_counts[f_old] -= 1
                fac_counts[f_new] += 1
                if not is_open[f_new]:
                    is_open[f_new] = True
                if fac_counts[f_old] == 0:
                    is_open[f_old] = False
                current_cost += best_move_delta
                improved = True
                passes = 0
                continue

            best_swap_delta = 0
            best_swap_i1 = -1
            best_swap_i2 = -1
            
            for i1 in range(n_cust):
                f1 = result[i1]
                for i2 in range(i1 + 1, n_cust):
                    f2 = result[i2]
                    if f1 == f2:
                        continue
                    if load[f1] - demands[i1] + demands[i2] > capacities[f1]:
                        continue
                    if load[f2] - demands[i2] + demands[i1] > capacities[f2]:
                        continue
                    delta = (assign_cost[i1, f2] + assign_cost[i2, f1]) - (assign_cost[i1, f1] + assign_cost[i2, f2])
                    if delta < best_swap_delta:
                        best_swap_delta = delta
                        best_swap_i1 = i1
                        best_swap_i2 = i2
                        
            if best_swap_i1 != -1:
                i1 = best_swap_i1
                i2 = best_swap_i2
                f1 = result[i1]
                f2 = result[i2]
                result[i1] = f2
                result[i2] = f1
                load[f1] += demands[i2] - demands[i1]
                load[f2] += demands[i1] - demands[i2]
                current_cost += best_swap_delta
                improved = True
                passes = 0

        return result, is_open, current_cost

    best_result = None
    best_cost = float('inf')
    
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        res, op = greedy_init()
        if res is None:
            continue
        res, op, cost = local_search(res, op)
        if cost < best_cost:
            best_cost = cost
            best_result = res.copy()
            
    if best_result is not None:
        return best_result
    else:
        # Emergency
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            feasible = np.where(capacities - load >= demands[i])[0]
            if len(feasible) == 0:
                f = np.argmax(capacities - load)
            else:
                costs = assign_cost[i, feasible]
                f = feasible[np.argmin(costs)]
            result[i] = f
            load[f] += demands[i]
        return result

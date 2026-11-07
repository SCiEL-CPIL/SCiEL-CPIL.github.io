import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    N = len(demands)
    M = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # 1. Greedy Initialization
    order = np.argsort(-demands)
    current_solution = np.zeros(N, dtype=int)
    load = np.zeros(M)
    
    for i in order:
        feasible_mask = load + demands[i] <= capacities
        if not np.any(feasible_mask):
            idx = np.argmax(capacities - load)
            current_solution[i] = idx
            load[idx] += demands[i]
            continue
            
        costs = assign_cost[i].copy()
        costs[~feasible_mask] = np.inf
        f = np.argmin(costs)
        current_solution[i] = f
        load[f] += demands[i]
        
    def calc_cost(sol, loads):
        open_mask = loads > 0
        cost = np.sum(open_cost[open_mask])
        assign_sum = np.sum([assign_cost[i, sol[i]] for i in range(N)])
        return cost + assign_sum

    current_cost = calc_cost(current_solution, load)
    best_solution = current_solution.copy()
    best_cost = current_cost
    
    # 2. Min-Cost Flow Formulation with Big-M for Fixed Costs
    # Network: Source -> Customer nodes -> Facility nodes -> Sink
    # Customer i has supply = demands[i]
    # Facility j has demand = capacities[j] (can send up to capacity)
    # Arc (i,j) has cost = assign_cost[i,j] + big_M * y_j
    # But we can't have y_j in a linear flow. Instead, use the standard trick:
    # Add a "fixed cost" arc from a dummy node or use the fact that if any flow
    # goes to facility j, we pay open_cost[j].
    # 
    # Better approach: Use the LP relaxation directly on the assignment variables
    # but with a cleaner formulation. Let's use the standard CFLP LP:
    # Min sum_{i,j} c_ij x_ij + sum_j F_j y_j
    # s.t. sum_j x_ij = d_i for all i
    #      sum_i x_ij <= cap_j for all j
    #      x_ij <= d_i * y_j for all i,j
    #      0 <= x_ij, 0 <= y_j <= 1
    #
    # This is the same as before but let's implement it more efficiently.
    # Actually, for min-cost flow with fixed charges, the standard technique is:
    # Create a network where each facility has two arcs to the sink:
    #   Arc A: capacity 0, cost open_cost[j] (the "activation" arc)
    #   Arc B: capacity cap_j, cost 0 (the "flow" arc)
    # And we need to ensure that if any flow goes through facility j,
    # the activation arc is used. This requires a binary variable or
    # a big-M constraint, which brings us back to the LP.
    #
    # Let's just solve the LP relaxation efficiently.
    
    def solve_lp():
        n_vars = N * M + M  # x[i*M+j] and y[j]
        
        # Objective
        c = np.zeros(n_vars)
        for i in range(N):
            for j in range(M):
                c[i * M + j] = assign_cost[i, j]
        for j in range(M):
            c[N * M + j] = open_cost[j]
        
        # Constraints
        # Equality: sum_j x[i,j] = demands[i] for each i
        A_eq = np.zeros((N, n_vars))
        b_eq = np.zeros(N)
        for i in range(N):
            for j in range(M):
                A_eq[i, i * M + j] = 1.0
            b_eq[i] = demands[i]
        
        # Inequality: sum_i x[i,j] <= capacities[j] for each j
        # Inequality: x[i,j] - demands[i]*y[j] <= 0 for each i,j
        A_ub = np.zeros((M + N * M, n_vars))
        b_ub = np.zeros(M + N * M)
        
        # Capacity constraints
        for j in range(M):
            for i in range(N):
                A_ub[j, i * M + j] = 1.0
            b_ub[j] = capacities[j]
        
        # Linking constraints: x[i,j] <= demands[i] * y[j]
        row_idx = M
        for i in range(N):
            for j in range(M):
                A_ub[row_idx, i * M + j] = 1.0
                A_ub[row_idx, N * M + j] = -demands[i]
                b_ub[row_idx] = 0.0
                row_idx += 1
        
        # Bounds
        bounds = []
        for i in range(N):
            for j in range(M):
                bounds.append((0, demands[i]))
        for j in range(M):
            bounds.append((0, 1))
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                     bounds=bounds, method='highs')
        
        if res.status != 0:
            return None
            
        x = res.x[:N*M].reshape(N, M)
        y = res.x[N*M:]
        
        # Round to integer solution using greedy repair
        sol = np.zeros(N, dtype=int)
        loads = np.zeros(M)
        
        # Sort by demand descending
        order = np.argsort(-demands)
        
        for i in order:
            feasible = loads + demands[i] <= capacities
            if not np.any(feasible):
                feasible = np.ones(M, dtype=bool)
                
            # Score: prefer high LP flow, then low cost
            scores = np.full(M, -np.inf)
            for j in range(M):
                if feasible[j]:
                    scores[j] = x[i, j] * 1000.0 - assign_cost[i, j]
                    
            best_j = np.argmax(scores)
            if scores[best_j] == -np.inf:
                costs = assign_cost[i].copy()
                costs[~feasible] = np.inf
                best_j = np.argmin(costs)
                
            sol[i] = best_j
            loads[best_j] += demands[i]
            
        return sol, loads

    # Try LP if time permits
    if time.time() - start_time < deadline - 10:
        lp_res = solve_lp()
        if lp_res is not None:
            sol_lp, loads_lp = lp_res
            cost_lp = calc_cost(sol_lp, loads_lp)
            if cost_lp < best_cost:
                best_cost = cost_lp
                best_solution = sol_lp.copy()
                current_solution = sol_lp.copy()
                load = loads_lp.copy()
                current_cost = cost_lp

    # 3. Local Search
    while time.time() - start_time < deadline:
        improved = False
        
        # Single move: best improvement
        best_delta = 0
        best_move = None
        
        for i in range(N):
            curr_f = current_solution[i]
            for k in range(M):
                if k == curr_f:
                    continue
                if load[k] + demands[i] > capacities[k]:
                    continue
                
                d_assign = assign_cost[i, k] - assign_cost[i, curr_f]
                
                d_open = 0
                new_load_curr = load[curr_f] - demands[i]
                if new_load_curr == 0:
                    d_open -= open_cost[curr_f]
                
                if load[k] == 0:
                    d_open += open_cost[k]
                    
                total_delta = d_assign + d_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_move = (i, k)
                    
        if best_move is not None:
            i, k = best_move
            curr_f = current_solution[i]
            load[curr_f] -= demands[i]
            load[k] += demands[i]
            current_solution[i] = k
            current_cost += best_delta
            improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution.copy()
                
        # Swap: best improvement
        best_delta = 0
        best_swap = None
        for i in range(N):
            f_i = current_solution[i]
            for m in range(i + 1, N):
                f_m = current_solution[m]
                if f_i == f_m:
                    continue
                
                new_load_fi = load[f_i] - demands[i] + demands[m]
                new_load_fm = load[f_m] - demands[m] + demands[i]
                
                if new_load_fi > capacities[f_i] or new_load_fm > capacities[f_m]:
                    continue
                
                d_assign = (assign_cost[i, f_m] + assign_cost[m, f_i]) - (assign_cost[i, f_i] + assign_cost[m, f_m])
                
                d_open = 0
                if new_load_fi == 0:
                    d_open -= open_cost[f_i]
                if new_load_fm == 0:
                    d_open -= open_cost[f_m]
                
                total_delta = d_assign + d_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_swap = (i, m)
                    
        if best_swap is not None:
            i, m = best_swap
            f_i = current_solution[i]
            f_m = current_solution[m]
            
            load[f_i] -= demands[i]
            load[f_i] += demands[m]
            load[f_m] -= demands[m]
            load[f_m] += demands[i]
            
            current_solution[i] = f_m
            current_solution[m] = f_i
            
            current_cost += best_delta
            improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution.copy()
                
        if not improved:
            break
            
    return best_solution.astype(int)

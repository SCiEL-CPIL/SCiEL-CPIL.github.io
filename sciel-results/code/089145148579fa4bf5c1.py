"""Capacitated facility location solver with advanced local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _total_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost for a given assignment."""
    load = np.zeros(len(capacities))
    for i in range(len(demands)):
        load[result[i]] += demands[i]
    if np.any(load > capacities):
        return float('inf')
    open_facs = np.where(load > 0)[0]
    cost = np.sum(open_cost[open_facs])
    cost += np.sum(assign_cost[np.arange(len(demands)), result])
    return cost


def _greedy_construct(demands, capacities, open_cost, assign_cost, rng=None):
    """Greedy construction: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    order = np.argsort(-demands)
    
    for i in order:
        fac_order = np.argsort(assign_cost[i])
        placed = False
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    placed = True
                    break
            if not placed:
                result[i] = 0
    
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit=10.0):
    """Local search with reassignment, swap, and facility closing moves."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_result = result.copy()
    best_cost = _total_cost(best_result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return best_result
    
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[best_result[i]] += demands[i]
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Move 1: Reassign a customer to a different facility
        for i in range(n_cust):
            current_f = best_result[i]
            for f in range(n_fac):
                if f == current_f:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    old_cost = assign_cost[i, current_f]
                    new_cost = assign_cost[i, f]
                    
                    open_change = 0.0
                    if load[current_f] == demands[i]:
                        open_change -= open_cost[current_f]
                    if load[f] == 0:
                        open_change += open_cost[f]
                    
                    delta = (new_cost - old_cost) + open_change
                    if delta < -1e-9:
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        best_result[i] = f
                        best_cost += delta
                        improved = True
                        break
        
        # Move 2: Swap two customers between facilities
        for i in range(n_cust):
            if time.time() - start_time >= time_limit:
                break
            for j in range(i + 1, n_cust):
                fi = best_result[i]
                fj = best_result[j]
                if fi == fj:
                    continue
                if load[fi] - demands[i] + demands[j] <= capacities[fi] and \
                   load[fj] - demands[j] + demands[i] <= capacities[fj]:
                    old_cost = assign_cost[i, fi] + assign_cost[j, fj]
                    new_cost = assign_cost[i, fj] + assign_cost[j, fi]
                    delta = new_cost - old_cost
                    if delta < -1e-9:
                        load[fi] += demands[j] - demands[i]
                        load[fj] += demands[i] - demands[j]
                        best_result[i], best_result[j] = fj, fi
                        best_cost += delta
                        improved = True
        
        # Move 3: Try to close a facility by moving all its customers
        for f in range(n_fac):
            if time.time() - start_time >= time_limit:
                break
            if load[f] == 0:
                continue
            custs_at_f = [i for i in range(n_cust) if best_result[i] == f]
            if len(custs_at_f) == 0:
                continue
            
            temp_load = load.copy()
            temp_result = best_result.copy()
            feasible = True
            total_delta = 0.0
            
            for i in custs_at_f:
                best_f_new = -1
                best_delta = float('inf')
                for f2 in range(n_fac):
                    if f2 == f:
                        continue
                    if temp_load[f2] + demands[i] <= capacities[f2]:
                        delta_move = assign_cost[i, f2] - assign_cost[i, f]
                        if temp_load[f2] == 0:
                            delta_move += open_cost[f2]
                        if delta_move < best_delta:
                            best_delta = delta_move
                            best_f_new = f2
                if best_f_new == -1:
                    feasible = False
                    break
                temp_load[f] -= demands[i]
                temp_load[best_f_new] += demands[i]
                temp_result[i] = best_f_new
                total_delta += best_delta
            
            if feasible:
                total_delta -= open_cost[f]
                if total_delta < -1e-9:
                    load = temp_load
                    best_result = temp_result
                    best_cost += total_delta
                    improved = True
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 5:
                break
        else:
            no_improve_count = 0
    
    return best_result


def _milp_solve(demands, capacities, open_cost, assign_cost, time_limit=30.0):
    """Solve CFLP using MILP via scipy linprog with branch-and-bound style approach."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Decision variables:
    # x[i][f] = 1 if customer i is assigned to facility f
    # y[f] = 1 if facility f is open
    # Total variables: n_cust * n_fac + n_fac
    
    # Objective: minimize sum(f) open_cost[f]*y[f] + sum(i,f) assign_cost[i,f]*x[i,f]
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum(f) x[i,f] = 1 for all i
    # 2. Capacity: sum(i) demands[i]*x[i,f] <= capacities[f]*y[f] for all f
    # 3. x[i,f] <= y[f] for all i,f (only assign to open facilities)
    # 4. x[i,f] >= 0, y[f] >= 0
    # 5. x[i,f] binary, y[f] binary
    
    # For linprog, we need to handle binary variables. 
    # We'll use a relaxation approach: solve LP relaxation, then round.
    # Or use a simpler heuristic based on LP solution.
    
    # Actually, scipy linprog doesn't support integer constraints directly.
    # Let's use a different approach: solve the LP relaxation and use it to guide search.
    
    # Build the LP relaxation
    # Variables: x[i*n_fac + f] for i in range(n_cust), f in range(n_fac)
    #            y[f] for f in range(n_fac)
    # Total: n_cust * n_fac + n_fac
    
    n_vars = n_cust * n_fac + n_fac
    
    # Objective coefficients
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            c[i * n_fac + f] = assign_cost[i, f]
    for f in range(n_fac):
        c[n_cust * n_fac + f] = open_cost[f]
    
    # Equality constraints: sum(f) x[i,f] = 1 for each i
    A_eq = np.zeros((n_cust, n_vars))
    for i in range(n_cust):
        for f in range(n_fac):
            A_eq[i, i * n_fac + f] = 1.0
    b_eq = np.ones(n_cust)
    
    # Inequality constraints:
    # Capacity: sum(i) demands[i]*x[i,f] - capacities[f]*y[f] <= 0 for all f
    # x[i,f] - y[f] <= 0 for all i,f
    
    n_ineq = n_fac + n_cust * n_fac
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    # Capacity constraints
    for f in range(n_fac):
        for i in range(n_cust):
            A_ub[f, i * n_fac + f] = demands[i]
        A_ub[f, n_cust * n_fac + f] = -capacities[f]
        b_ub[f] = 0
    
    # x[i,f] <= y[f]
    idx = n_fac
    for i in range(n_cust):
        for f in range(n_fac):
            A_ub[idx, i * n_fac + f] = 1.0
            A_ub[idx, n_cust * n_fac + f] = -1.0
            b_ub[idx] = 0
            idx += 1
    
    # Bounds: 0 <= x[i,f] <= 1, 0 <= y[f] <= 1
    bounds = [(0, 1)] * n_vars
    
    # Solve LP relaxation
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status != 0:
            return None
    except:
        return None
    
    # Extract solution
    x_sol = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
    y_sol = res.x[n_cust * n_fac:]
    
    # Round to get integer solution: assign each customer to the facility with highest x[i,f]
    # that has y[f] > 0.5
    result = np.zeros(n_cust, dtype=int)
    for i in range(n_cust):
        # Prefer facilities with y[f] > 0.5 and highest x[i,f]
        candidates = np.where(y_sol > 0.3)[0]
        if len(candidates) == 0:
            candidates = np.arange(n_fac)
        # Among candidates, pick the one with highest x[i,f]
        best_f = candidates[np.argmax(x_sol[i, candidates])]
        result[i] = best_f
    
    # Check feasibility
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    if np.any(load > capacities):
        return None
    
    return result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities with multi-start local search and MILP guidance."""
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_result = None
    best_cost = float('inf')
    
    # First, try to get a good initial solution from LP relaxation
    if time.time() < deadline - 60.0:
        milp_result = _milp_solve(demands, capacities, open_cost, assign_cost, time_limit=20.0)
        if milp_result is not None:
            cost = _total_cost(milp_result, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = milp_result.copy()
    
    # Generate multiple diverse initial solutions
    rng = np.random.RandomState(42)
    
    # Strategy 1: Greedy with different randomizations
    for trial in range(5):
        if time.time() > deadline - 30.0:
            break
        
        # Perturb the assignment costs slightly for diversity
        if trial == 0:
            perturbed_cost = assign_cost
        else:
            noise = rng.random((n_cust, n_fac)) * 10.0
            perturbed_cost = assign_cost + noise
        
        result = _greedy_construct(demands, capacities, open_cost, perturbed_cost)
        
        # Local search
        time_left = deadline - time.time() - 5.0
        if time_left < 1.0:
            cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
            continue
        
        result = _local_search(result, demands, capacities, open_cost, assign_cost,
                              time_limit=min(time_left, 15.0))
        
        cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Final polish: additional local search iterations on the best solution
    if best_result is not None and time.time() < deadline - 10.0:
        time_left = deadline - time.time() - 5.0
        if time_left > 1.0:
            polished = _local_search(best_result, demands, capacities, open_cost, assign_cost,
                                    time_limit=time_left)
            cost = _total_cost(polished, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = polished.copy()
    
    if best_result is None:
        best_result = _greedy_construct(demands, capacities, open_cost, assign_cost)
    
    return best_result
# EVOLVE-BLOCK-END

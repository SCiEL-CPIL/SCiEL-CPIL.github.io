"""Capacitated facility location: MILP via HiGHS with local-search polish and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _HAS_SCIPY_MILP = True
except Exception:
    _HAS_SCIPY_MILP = False

try:
    import highspy
    _HAS_HIGHS = True
except Exception:
    _HAS_HIGHS = False

_DEADLINE_SEC = 115.0  # safety margin before 120s kill


def _greedy_feasible(demands, capacities, open_cost, assign_cost):
    """Largest-demand-first greedy assignment (always feasible)."""
    n = len(demands)
    m = len(capacities)
    result = np.zeros(n, dtype=int)
    load = np.zeros(m, dtype=float)
    order = np.argsort(-demands)
    for i in order:
        # cheapest facility that can still take this customer
        costs = assign_cost[i]
        feasible = load + demands[i] <= capacities
        # set infeasible to inf
        c = costs.copy()
        c[~feasible] = np.inf
        f = int(np.argmin(c))
        if not np.isfinite(c[f]):
            # fallback: any facility (should not happen if total demand <= total cap)
            f = int(np.argmin(costs))
        load[f] += demands[i]
        result[i] = f
    return result


def _total_cost(demands, capacities, open_cost, assign_cost, result):
    """Compute the objective: open costs + assignment costs."""
    n = len(demands)
    m = len(capacities)
    # assignment cost
    ac = float(np.sum(assign_cost[np.arange(n), result]))
    # open cost: facilities with at least one customer
    used = np.zeros(m, dtype=bool)
    for f in result:
        used[f] = True
    oc = float(np.sum(open_cost[used]))
    return ac + oc


def _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=5.0):
    """Move-based local search: try reassigning one customer to a different facility.
    Accept if total cost decreases and capacity constraints hold."""
    n = len(demands)
    m = len(capacities)
    result = result.copy().astype(int)
    load = np.zeros(m, dtype=float)
    for i in range(n):
        load[result[i]] += demands[i]
    
    def used_mask(res):
        mask = np.zeros(m, dtype=bool)
        for f in res:
            mask[f] = True
        return mask
    
    def obj(res):
        ac = float(np.sum(assign_cost[np.arange(n), res]))
        um = used_mask(res)
        oc = float(np.sum(open_cost[um]))
        return ac + oc
    
    best = obj(result)
    t_start = time.time()
    improved = True
    while improved and (time.time() - t_start) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - t_start) > time_limit:
                break
            cur_f = result[i]
            cur_load = load
            # try each other facility
            for f in range(m):
                if f == cur_f:
                    continue
                # capacity check
                if cur_load[f] + demands[i] > capacities[f]:
                    continue
                # compute delta in assignment cost
                delta_ac = assign_cost[i, f] - assign_cost[i, cur_f]
                # open cost change:
                # if cur_f becomes empty and f was empty -> +open_cost[cur_f]
                # if cur_f was empty (no, it has i) and f becomes non-empty -> +open_cost[f]
                # etc.
                old_used = used_mask(result)
                # simulate
                old_count_cur = int(np.sum(result == cur_f))
                old_count_f = int(np.sum(result == f))
                
                # new open cost delta
                delta_oc = 0.0
                if old_count_f == 0:
                    delta_oc += open_cost[f]  # opening f
                if old_count_cur == 1:
                    delta_oc -= open_cost[cur_f]  # closing cur_f
                
                delta = delta_ac + delta_oc
                if delta < -1e-9:
                    # accept
                    load[cur_f] -= demands[i]
                    load[f] += demands[i]
                    result[i] = f
                    best += delta
                    improved = True
                    break  # restart scan after move
    return result


def _solve_milp(demands, capacities, open_cost, assign_cost, time_limit=10.0):
    """Solve the capacitated facility location MILP using scipy.optimize.milp."""
    n = len(demands)
    m = len(capacities)
    
    # Variables: x[i,j] for i in [0,n), j in [0,m) -> n*m variables
    # y[j] for j in [0,m) -> m variables
    # Total vars: n*m + m
    nm = n * m
    total_vars = nm + m
    
    # Objective: sum_{i,j} assign_cost[i,j] * x[i,j] + sum_j open_cost[j] * y[j]
    c = np.zeros(total_vars)
    for i in range(n):
        for j in range(m):
            c[i * m + j] = assign_cost[i, j]
    for j in range(m):
        c[nm + j] = open_cost[j]
    
    # Constraints:
    # 1) For each customer i: sum_j x[i,j] = 1  (n equality constraints)
    # 2) For each facility j: sum_i demands[i] * x[i,j] <= capacities[j] * y[j]  (m constraints)
    #    Rearranged: sum_i demands[i] * x[i,j] - capacities[j] * y[j] <= 0
    # 3) x[i,j] in {0,1}, y[j] in {0,1}
    
    # Build A matrix for constraints
    # Constraint type: equality for customer assignment, inequality for capacity
    num_cons = n + m
    A = np.zeros((num_cons, total_vars))
    lb = np.zeros(num_cons)
    ub = np.zeros(num_cons)
    
    for i in range(n):
        for j in range(m):
            A[i, i * m + j] = 1.0
        lb[i] = 1.0
        ub[i] = 1.0
    
    for j in range(m):
        for i in range(n):
            A[n + j, i * m + j] = demands[i]
        A[n + j, nm + j] = -capacities[j]
        lb[n + j] = -np.inf
        ub[n + j] = 0.0
    
    # Integrality: x and y are binary -> integrality 1
    integrality = np.ones(total_vars)
    
    # Bounds: 0 <= var <= 1
    var_lb = np.zeros(total_vars)
    var_ub = np.ones(total_vars)
    
    constraint = LinearConstraint(A, lb, ub)
    bounds = Bounds(var_lb, var_ub)
    
    res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds,
               options={'time_limit': time_limit, 'mip_rel_gap': 0.05})
    
    if res is None or not res.success or res.x is None:
        return None
    
    x = res.x[:nm].reshape(n, m)
    # Round to nearest integer (should already be near 0/1)
    x_rounded = np.round(x).astype(int)
    
    # Extract assignment
    result = np.zeros(n, dtype=int)
    for i in range(n):
        # find j with x[i,j] = 1
        js = np.where(x_rounded[i] == 1)[0]
        if len(js) == 0:
            # fallback
            result[i] = int(np.argmin(assign_cost[i]))
        else:
            result[i] = int(js[0])
    
    # Verify feasibility
    load = np.zeros(m, dtype=float)
    for i in range(n):
        load[result[i]] += demands[i]
    if np.any(load > capacities + 1e-6):
        return None
    
    return result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign each customer to a facility minimizing open + assignment costs subject to capacity."""
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n = len(demands)
    m = len(capacities)
    t_start = time.time()
    
    # Always start with a feasible greedy solution
    best_result = _greedy_feasible(demands, capacities, open_cost, assign_cost)
    best_cost = _total_cost(demands, capacities, open_cost, assign_cost, best_result)
    
    # Try MILP if available and time permits
    if _HAS_SCIPY_MILP:
        elapsed = time.time() - t_start
        remaining = _DEADLINE_SEC - elapsed - 5.0  # leave 5s for local search + safety
        if remaining > 2.0:
            try:
                milp_result = _solve_milp(demands, capacities, open_cost, assign_cost,
                                          time_limit=remaining)
                if milp_result is not None:
                    milp_cost = _total_cost(demands, capacities, open_cost, assign_cost, milp_result)
                    if milp_cost < best_cost:
                        best_result = milp_result
                        best_cost = milp_cost
            except Exception:
                pass
    
    # Local search polish if time remains
    elapsed = time.time() - t_start
    remaining = _DEADLINE_SEC - elapsed - 2.0
    if remaining > 1.0:
        try:
            ls_result = _local_search(demands, capacities, open_cost, assign_cost,
                                       best_result, time_limit=remaining)
            ls_cost = _total_cost(demands, capacities, open_cost, assign_cost, ls_result)
            if ls_cost < best_cost:
                best_result = ls_result
                best_cost = ls_cost
        except Exception:
            pass
    
    # Final feasibility check
    load = np.zeros(m, dtype=float)
    for i in range(n):
        f = int(best_result[i])
        if f < 0 or f >= m:
            # repair
            best_result[i] = int(np.argmin(assign_cost[i]))
            f = int(best_result[i])
        load[f] += demands[i]
    
    if np.any(load > capacities + 1e-6):
        # Fallback to pure greedy
        best_result = _greedy_feasible(demands, capacities, open_cost, assign_cost)
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END

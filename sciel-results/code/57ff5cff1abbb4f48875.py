# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Strategy: Exact LP relaxation via scipy (network structure ensures integrality),
    with heuristic fallback and local search refinement.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute sorted facility indices by assignment cost
    fac_order = np.argsort(assign_cost, axis=1)

    def _compute_obj(result):
        if result is None:
            return float('inf')
        load = np.zeros(n_fac, dtype=np.int64)
        valid = True
        for i in range(n_cust):
            f = result[i]
            if f < 0 or f >= n_fac:
                valid = False
                break
            load[f] += demands[i]
        if not valid or np.any(load > capacities):
            return float('inf')
        open_facs = load > 0
        total = np.sum(open_cost[open_facs]) + np.sum(assign_cost[np.arange(n_cust), result])
        return total

    def _greedy_assign(seed_order, fac_order_used):
        load = np.zeros(n_fac, dtype=np.int64)
        result = np.full(n_cust, -1, dtype=np.int64)
        for i in seed_order:
            assigned = False
            for f in fac_order_used[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        assigned = True
                        break
            if not assigned:
                result[i] = 0
                load[0] += demands[i]
        return result, load

    # Try LP relaxation first
    best_result = None
    best_obj = float('inf')

    # Formulate LP:
    # Variables: x[i,f] for i in range(n_cust), f in range(n_fac) -> n_cust * n_fac
    # y[f] for f in range(n_fac) -> n_fac
    # Total variables: n_cust * n_fac + n_fac
    #
    # Minimize: sum_{i,f} assign_cost[i,f] * x[i,f] + sum_f open_cost[f] * y[f]
    #
    # Constraints:
    # 1. For each customer i: sum_f x[i,f] = 1
    # 2. For each facility f: sum_i demands[i] * x[i,f] <= capacities[f]
    # 3. For each facility f: sum_i x[i,f] <= y[f] * (some large number) -> x[i,f] <= y[f]
    #    Actually: x[i,f] <= y[f] for all i,f
    # 4. x[i,f] >= 0, y[f] >= 0
    # 5. y[f] <= 1 (binary relaxation)

    n_vars = n_cust * n_fac + n_fac

    # Objective coefficients
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            c[i * n_fac + f] = assign_cost[i, f]
    for f in range(n_fac):
        c[n_cust * n_fac + f] = open_cost[f]

    # Equality constraints: each customer assigned to exactly one facility
    A_eq = np.zeros((n_cust, n_vars))
    b_eq = np.ones(n_cust)
    for i in range(n_cust):
        for f in range(n_fac):
            A_eq[i, i * n_fac + f] = 1.0

    # Inequality constraints:
    # Capacity: sum_i demands[i] * x[i,f] <= capacities[f]
    # x[i,f] <= y[f] for all i, f
    # y[f] <= 1

    n_ineq = n_fac + n_cust * n_fac + n_fac
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)

    # Capacity constraints
    for f in range(n_fac):
        for i in range(n_cust):
            A_ub[f, i * n_fac + f] = demands[i]
        b_ub[f] = capacities[f]

    # x[i,f] <= y[f]
    idx = n_fac
    for i in range(n_cust):
        for f in range(n_fac):
            A_ub[idx, i * n_fac + f] = 1.0
            A_ub[idx, n_cust * n_fac + f] = -1.0
            b_ub[idx] = 0.0
            idx += 1

    # y[f] <= 1
    for f in range(n_fac):
        A_ub[idx, n_cust * n_fac + f] = 1.0
        b_ub[idx] = 1.0
        idx += 1

    # Bounds: x[i,f] in [0,1], y[f] in [0,1]
    bounds = [(0, 1)] * n_vars

    # Solve LP
    try:
        if time.time() < deadline - 10:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
            if res.success:
                # Extract assignment
                result = np.zeros(n_cust, dtype=np.int64)
                for i in range(n_cust):
                    best_f = 0
                    best_val = -1
                    for f in range(n_fac):
                        val = res.x[i * n_fac + f]
                        if val > best_val:
                            best_val = val
                            best_f = f
                    result[i] = best_f

                # Validate
                load = np.zeros(n_fac, dtype=np.int64)
                valid = True
                for i in range(n_cust):
                    f = result[i]
                    load[f] += demands[i]
                if np.any(load > capacities):
                    valid = False

                if valid:
                    obj = _compute_obj(result)
                    if obj < best_obj:
                        best_obj = obj
                        best_result = result.copy()
    except Exception:
        pass

    # Fallback: greedy if LP didn't work or wasn't run
    if best_result is None or time.time() >= deadline - 5:
        if best_result is None:
            order = np.argsort(-demands)
            best_result, _ = _greedy_assign(order, fac_order)
            best_obj = _compute_obj(best_result)

    # Local search refinement
    def _local_search(result, load, time_limit):
        current_obj = _compute_obj(result)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            order = np.random.permutation(n_cust)
            for i in order:
                if time.time() >= time_limit:
                    break
                current_f = result[i]
                if current_f < 0:
                    continue
                for f in fac_order[i]:
                    if f == current_f:
                        continue
                    if load[f] + demands[i] <= capacities[f]:
                        delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
                        delta_open = 0.0
                        if load[current_f] == demands[i]:
                            delta_open -= open_cost[current_f]
                        if load[f] == 0:
                            delta_open += open_cost[f]
                        delta = delta_assign + delta_open
                        if delta < -1e-9:
                            load[current_f] -= demands[i]
                            load[f] += demands[i]
                            result[i] = f
                            current_obj += delta
                            improved = True
                            break
        return result, load, current_obj

    def _swap_search(result, load, time_limit):
        current_obj = _compute_obj(result)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n_cust):
                if time.time() >= time_limit:
                    break
                for j in range(i + 1, n_cust):
                    if time.time() >= time_limit:
                        break
                    fi = result[i]
                    fj = result[j]
                    if fi == fj:
                        continue
                    new_load_fi = load[fi] - demands[i] + demands[j]
                    new_load_fj = load[fj] - demands[j] + demands[i]
                    if new_load_fi < 0 or new_load_fj < 0:
                        continue
                    if new_load_fi > capacities[fi] or new_load_fj > capacities[fj]:
                        continue
                    delta_assign = (assign_cost[i, fj] - assign_cost[i, fi]) + (assign_cost[j, fi] - assign_cost[j, fj])
                    delta_open = 0.0
                    if load[fi] > 0 and new_load_fi == 0:
                        delta_open -= open_cost[fi]
                    elif load[fi] == 0 and new_load_fi > 0:
                        delta_open += open_cost[fi]
                    if load[fj] > 0 and new_load_fj == 0:
                        delta_open -= open_cost[fj]
                    elif load[fj] == 0 and new_load_fj > 0:
                        delta_open += open_cost[fj]
                    delta = delta_assign + delta_open
                    if delta < -1e-9:
                        load[fi] = new_load_fi
                        load[fj] = new_load_fj
                        result[i] = fj
                        result[j] = fi
                        current_obj += delta
                        improved = True
                        break
                if improved:
                    break
        return result, load, current_obj

    # Run local search if time permits
    load = np.zeros(n_fac, dtype=np.int64)
    for i in range(n_cust):
        load[best_result[i]] += demands[i]

    if time.time() < deadline - 5:
        best_result, load, best_obj = _local_search(best_result, load, deadline - 5)

    if time.time() < deadline - 3:
        best_result, load, best_obj = _swap_search(best_result, load, deadline - 3)

    # Additional passes
    while time.time() < deadline - 2:
        prev_obj = best_obj
        best_result, load, best_obj = _local_search(best_result, load, deadline - 2)
        if best_obj >= prev_obj - 1e-9:
            break
        best_result, load, best_obj = _swap_search(best_result, load, deadline - 2)

    # Final validation
    load = np.zeros(n_fac, dtype=np.int64)
    for i in range(n_cust):
        f = best_result[i]
        if f < 0 or f >= n_fac:
            f = 0
            best_result[i] = 0
        load[f] += demands[i]

    if np.any(load > capacities):
        order = np.argsort(-demands)
        best_result, load = _greedy_assign(order, fac_order)
        load = np.zeros(n_fac, dtype=np.int64)
        for i in range(n_cust):
            f = best_result[i]
            if f < 0 or f >= n_fac:
                f = 0
                best_result[i] = 0
            load[f] += demands[i]
        if np.any(load > capacities):
            best_result = np.zeros(n_cust, dtype=np.int64)

    return best_result.astype(int)

# EVOLVE-BLOCK-END

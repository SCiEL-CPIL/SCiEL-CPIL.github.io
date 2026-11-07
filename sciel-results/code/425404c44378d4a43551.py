"""Capacitated facility location via LP relaxation with HiGHS solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n = len(demands)
    m = len(capacities)

    # Fallback greedy solution (largest demand first, cheapest feasible)
    def _greedy():
        result = np.zeros(n, dtype=int)
        load = np.zeros(m)
        for i in np.argsort(-demands):
            placed = False
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    result[i] = f
                    placed = True
                    break
            if not placed:
                # Force assign to least-loaded facility
                f = np.argmin(load)
                load[f] += demands[i]
                result[i] = f
        return result

    greedy_result = _greedy()

    # Formulate LP:
    # Variables: x[0..n*m-1] for assignments, y[0..m-1] for facility opening
    # Total variables: n*m + m
    #
    # Objective: sum_f open_cost[f]*y[f] + sum_c,f assign_cost[c,f]*x[c,f]
    #
    # Constraints:
    # 1) For each customer c: sum_f x[c,f] = 1  (n equality constraints)
    # 2) For each facility f: sum_c demands[c]*x[c,f] - capacities[f]*y[f] <= 0  (m inequality constraints)
    #
    # Bounds: x >= 0, 0 <= y <= 1

    nm = n * m
    nvars = nm + m

    # Objective vector
    c_obj = np.zeros(nvars)
    c_obj[:nm] = assign_cost.ravel()
    c_obj[nm:] = open_cost

    # Equality constraints: A_eq @ vars = b_eq
    # For each customer c, sum over f of x[c,f] = 1
    A_eq = np.zeros((n, nvars))
    for c in range(n):
        for f in range(m):
            A_eq[c, c * m + f] = 1.0
    b_eq = np.ones(n)

    # Inequality constraints: A_ub @ vars <= b_ub
    # For each facility f: sum_c demands[c]*x[c,f] - capacities[f]*y[f] <= 0
    A_ub = np.zeros((m, nvars))
    for f in range(m):
        for c in range(n):
            A_ub[f, c * m + f] = demands[c]
        A_ub[f, nm + f] = -capacities[f]
    b_ub = np.zeros(m)

    # Bounds
    bounds = [(0, None)] * nm + [(0, 1)] * m

    # Solve LP
    try:
        res = linprog(
            c_obj,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'time_limit': DEADLINE - time.time() - 5.0}
        )

        if res.status == 0 and res.x is not None:
            x_vals = res.x[:nm].reshape(n, m)
            # Round assignments: for each customer, pick facility with max x value
            result = np.zeros(n, dtype=int)
            for c in range(n):
                # Check if solution is close to integral
                best_f = int(np.argmax(x_vals[c]))
                result[c] = best_f

            # Verify capacity constraints after rounding
            load = np.zeros(m)
            for c in range(n):
                load[result[c]] += demands[c]

            # If any capacity violated, fall back to greedy
            if np.all(load <= capacities + 1e-6):
                return result
            else:
                # Try to fix by reassigning over-capacity customers
                for f in range(m):
                    if load[f] > capacities[f] + 1e-6:
                        # Find customers assigned to f, reassign to cheapest alternative
                        excess = load[f] - capacities[f]
                        custs_at_f = np.where(result == f)[0]
                        # Sort by demand descending, reassign largest first
                        custs_at_f = custs_at_f[np.argsort(-demands[custs_at_f])]
                        for c in custs_at_f:
                            if excess <= 0:
                                break
                            # Find cheapest facility with remaining capacity
                            for f2 in np.argsort(assign_cost[c]):
                                if f2 == f:
                                    continue
                                if load[f2] + demands[c] <= capacities[f2] + 1e-9:
                                    load[f] -= demands[c]
                                    load[f2] += demands[c]
                                    result[c] = f2
                                    excess -= demands[c]
                                    break
                # Final check
                if np.all(load <= capacities + 1e-6):
                    return result

        # Check time
        if time.time() > DEADLINE - 10.0:
            return greedy_result

    except Exception:
        pass

    # Fallback to greedy
    return greedy_result
# EVOLVE-BLOCK-END

"""Rewritten solver for capacitated facility location using MILP + local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DEADLINE_MARGIN = 10.0  # seconds of safety margin before the 150s limit


def _solve_milp(demands, capacities, open_cost, assign_cost):
    """Solve the capacitated facility location problem via MILP."""
    n_cust = len(demands)
    n_fac = len(capacities)

    # Variables: x[c,f] (binary) for c in range(n_cust), f in range(n_fac)
    #            y[f] (binary) for f in range(n_fac)
    # Total variables: n_cust * n_fac + n_fac
    n_vars = n_cust * n_fac + n_fac

    # Objective: minimize sum_c,f assign_cost[c,f] * x[c,f] + sum_f open_cost[f] * y[f]
    c_obj = np.zeros(n_vars)
    for c in range(n_cust):
        for f in range(n_fac):
            c_obj[c * n_fac + f] = assign_cost[c, f]
    for f in range(n_fac):
        c_obj[n_cust * n_fac + f] = open_cost[f]

    # Constraints:
    # 1) Each customer assigned to exactly one facility: sum_f x[c,f] = 1 for all c
    # 2) Capacity: sum_c demands[c] * x[c,f] <= capacities[f] for all f
    # 3) Linking: x[c,f] <= y[f] for all c,f (facility must be open to serve)

    A_rows = []
    lb = []
    ub = []

    # Constraint 1: assignment (n_cust rows)
    for c in range(n_cust):
        row = np.zeros(n_vars)
        for f in range(n_fac):
            row[c * n_fac + f] = 1.0
        A_rows.append(row)
        lb.append(1.0)
        ub.append(1.0)

    # Constraint 2: capacity (n_fac rows)
    for f in range(n_fac):
        row = np.zeros(n_vars)
        for c in range(n_cust):
            row[c * n_fac + f] = float(demands[c])
        A_rows.append(row)
        lb.append(0.0)
        ub.append(float(capacities[f]))

    # Constraint 3: linking (n_cust * n_fac rows)
    for c in range(n_cust):
        for f in range(n_fac):
            row = np.zeros(n_vars)
            row[c * n_fac + f] = 1.0
            row[n_cust * n_fac + f] = -1.0
            A_rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)

    A_mat = np.array(A_rows)

    # Variable bounds and integrality
    integrality = np.ones(n_vars)  # all binary
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

    constraint = LinearConstraint(A_mat, lb, ub)

    try:
        res = milp(
            c=c_obj,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 120, "mip_rel_gap": 0.05}
        )
    except Exception:
        return None

    if res.status != 0:  # 0 means optimal or gap reached
        # Still might have a feasible solution
        if res.x is None:
            return None

    x_vals = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
    # Find best facility for each customer (largest x value)
    result = np.argmax(x_vals, axis=1)

    # Verify feasibility
    load = np.zeros(n_fac)
    for c in range(n_cust):
        load[result[c]] += demands[c]
    if np.any(load > capacities + 1e-6):
        return None
    if np.any(result < 0) or np.any(result >= n_fac):
        return None
    if np.any(np.isnan(result)):
        return None

    return result.astype(int)


def _greedy_assign(demands, capacities, open_cost, assign_cost):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)

    order = np.argsort(-demands)
    for c in order:
        best_f = -1
        best_cost = np.inf
        for f in range(n_fac):
            if load[f] + demands[c] <= capacities[f]:
                # Cost includes potential open cost
                marginal = assign_cost[c, f]
                if load[f] == 0:
                    marginal += open_cost[f]
                if marginal < best_cost:
                    best_cost = marginal
                    best_f = f
        if best_f == -1:
            # Shouldn't happen if total capacity >= total demand
            best_f = 0
        load[best_f] += demands[c]
        result[c] = best_f

    return result


def _local_search(demands, capacities, open_cost, assign_cost, result, max_iters=500):
    """Local search: try reassigning customers to reduce cost."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy().astype(int)
    load = np.zeros(n_fac)
    for c in range(n_cust):
        load[result[c]] += demands[c]

    def total_cost():
        cost = 0.0
        for c in range(n_cust):
            cost += assign_cost[c, result[c]]
        for f in range(n_fac):
            if load[f] > 0:
                cost += open_cost[f]
        return cost

    best_cost = total_cost()
    no_improve = 0

    for _ in range(max_iters):
        improved = False
        for c in range(n_cust):
            cur_f = result[c]
            cur_cost = total_cost()
            for new_f in range(n_fac):
                if new_f == cur_f:
                    continue
                if load[new_f] + demands[c] > capacities[new_f]:
                    continue
                # Compute cost delta
                delta = assign_cost[c, new_f] - assign_cost[c, cur_f]
                # Open cost changes
                if load[cur_f] == demands[c]:
                    # Closing cur_f
                    delta -= open_cost[cur_f]
                if load[new_f] == 0:
                    # Opening new_f
                    delta += open_cost[new_f]

                if delta < -1e-9:
                    # Apply move
                    load[cur_f] -= demands[c]
                    load[new_f] += demands[c]
                    result[c] = new_f
                    best_cost += delta
                    improved = True
                    break
        if not improved:
            break

    return result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign each customer to a facility, minimizing total cost."""
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    # Get a feasible solution via greedy
    result = _greedy_assign(demands, capacities, open_cost, assign_cost)

    # Try MILP if we have time
    if time.time() < deadline - 30:
        milp_result = _solve_milp(demands, capacities, open_cost, assign_cost)
        if milp_result is not None:
            # Verify MILP result is valid
            load = np.zeros(n_fac)
            for c in range(n_cust):
                load[milp_result[c]] += demands[c]
            if np.all(load <= capacities + 1e-6):
                result = milp_result

    # Local search refinement if time permits
    if time.time() < deadline - 5:
        result = _local_search(demands, capacities, open_cost, assign_cost, result)

    # Final safety check
    result = result.astype(int)
    load = np.zeros(n_fac)
    for c in range(n_cust):
        load[result[c]] += demands[c]
    if np.any(load > capacities + 1e-6):
        # Fallback to greedy
        result = _greedy_assign(demands, capacities, open_cost, assign_cost)

    return result
# EVOLVE-BLOCK-END

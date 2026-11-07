"""Capacitated facility location: LP relaxation (HiGHS) + greedy repair + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign customers to facilities using:
    1. LP relaxation solved via scipy.optimize.linprog (HiGHS) to get a strong initial solution
    2. Greedy repair for feasibility
    3. Local search (moves and swaps) to reduce cost
    """
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def total_cost(assignment):
        """Compute total cost: open cost + assignment cost."""
        used = np.zeros(n_facilities, dtype=bool)
        for f in assignment:
            used[f] = True
        oc = open_cost[used].sum()
        ac = assign_cost[np.arange(n_customers), assignment].sum()
        return oc + ac

    def is_feasible(assignment):
        load = np.zeros(n_facilities, dtype=np.float64)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        return np.all(load <= capacities + 1e-9)

    def greedy_init():
        """Greedy initialization: sort by demand descending, assign to cheapest feasible facility."""
        order = np.argsort(-demands)
        load = np.zeros(n_facilities, dtype=np.float64)
        assignment = np.zeros(n_customers, dtype=int)
        for i in order:
            placed = False
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    assignment[i] = f
                    placed = True
                    break
            if not placed:
                remaining = capacities - load
                f = int(np.argmax(remaining))
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    assignment[i] = f
                else:
                    assignment[i] = f
                    load[f] += demands[i]
        return assignment

    def solve_lp():
        """
        Solve the LP relaxation of the facility location problem.
        Variables: x[i,f] for customer i assigned to facility f (fractional)
                   y[f] for facility f open (fractional)
        """
        n_x = n_customers * n_facilities
        n_y = n_facilities
        n_vars = n_x + n_y

        c = np.concatenate([assign_cost.flatten(), open_cost])

        A_eq = np.zeros((n_customers, n_vars))
        b_eq = np.ones(n_customers)
        for i in range(n_customers):
            for f in range(n_facilities):
                A_eq[i, i * n_facilities + f] = 1.0

        A_ub = np.zeros((n_facilities, n_vars))
        b_ub = np.zeros(n_facilities)
        for f in range(n_facilities):
            for i in range(n_customers):
                A_ub[f, i * n_facilities + f] = demands[i]
            A_ub[f, n_x + f] = -capacities[f]

        bounds = [(0, 1)] * n_vars

        if time.time() > DEADLINE - 60:
            return None

        try:
            res = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs'
            )

            if res.success:
                x_vals = res.x[:n_x].reshape(n_customers, n_facilities)
                assignment = np.argmax(x_vals, axis=1).astype(int)
                return assignment
            else:
                return None
        except Exception:
            return None

    lp_assignment = solve_lp()

    if lp_assignment is not None and is_feasible(lp_assignment):
        result = lp_assignment.copy()
    else:
        result = greedy_init()

    def local_search(assignment, max_time=120.0):
        """Improve assignment via moves and swaps."""
        assignment = assignment.copy().astype(int)
        load = np.zeros(n_facilities, dtype=np.float64)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]

        used = np.zeros(n_facilities, dtype=bool)
        for f in assignment:
            used[f] = True
        current_cost = open_cost[used].sum() + assign_cost[np.arange(n_customers), assignment].sum()

        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n_customers):
                if time.time() > max_time:
                    break
                old_f = assignment[i]
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[i] > capacities[new_f] + 1e-9:
                        continue

                    old_load_after = load[old_f] - demands[i]
                    new_load_before = load[new_f]

                    oc_delta = 0.0
                    if old_load_after <= 1e-9 and used[old_f]:
                        oc_delta -= open_cost[old_f]
                    if new_load_before <= 1e-9 and not used[new_f]:
                        oc_delta += open_cost[new_f]

                    ac_delta = assign_cost[i, new_f] - assign_cost[i, old_f]

                    delta = oc_delta + ac_delta
                    if delta < -1e-9:
                        load[old_f] = old_load_after
                        load[new_f] = new_load_before + demands[i]
                        if old_load_after <= 1e-9:
                            used[old_f] = False
                        if new_load_before <= 1e-9:
                            used[new_f] = True
                        assignment[i] = new_f
                        current_cost += delta
                        improved = True
                        break

            if time.time() > max_time:
                break
            for i in range(n_customers):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n_customers):
                    if time.time() > max_time:
                        break
                    fi = assignment[i]
                    fj = assignment[j]
                    if fi == fj:
                        continue

                    new_load_fi = load[fi] - demands[i] + demands[j]
                    new_load_fj = load[fj] - demands[j] + demands[i]

                    if new_load_fi > capacities[fi] + 1e-9:
                        continue
                    if new_load_fj > capacities[fj] + 1e-9:
                        continue

                    oc_delta = 0.0
                    ac_delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])

                    delta = oc_delta + ac_delta
                    if delta < -1e-9:
                        load[fi] = new_load_fi
                        load[fj] = new_load_fj
                        assignment[i] = fj
                        assignment[j] = fi
                        current_cost += delta
                        improved = True
                        break

        return assignment, current_cost

    remaining_time = DEADLINE - time.time()
    if remaining_time > 5:
        result, _ = local_search(result, max_time=min(remaining_time, 120.0))

    if not is_feasible(result):
        result = greedy_init()

    return result.astype(int)
# EVOLVE-BLOCK-END

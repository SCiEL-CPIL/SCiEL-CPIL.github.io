# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location solver.
    Strategy:
      1. Greedy initial assignment.
      2. Solve min-cost flow (LP) on a candidate open set to get a strong baseline.
      3. Iteratively try to close facilities (drop the worst open facility) and re-solve flow.
      4. Local search: move customers to reduce cost while respecting capacities.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_c = len(demands)
    n_f = len(capacities)

    def total_cost(assignment):
        load = np.zeros(n_f)
        for c in range(n_c):
            f = int(assignment[c])
            load[f] += demands[c]
        if np.any(load > capacities + 1e-9):
            return np.inf
        cost = 0.0
        used = np.zeros(n_f, dtype=bool)
        for c in range(n_c):
            f = int(assignment[c])
            cost += assign_cost[c, f]
            used[f] = True
        cost += open_cost[used].sum()
        return cost

    def solve_flow(open_mask, max_time=10.0):
        """
        Solve the transportation problem: assign each customer to an open facility
        minimizing total assignment cost, subject to capacity constraints.
        Returns (assignment, cost) or (None, None) if infeasible or timeout.
        """
        t0 = time.time()
        open_idx = np.where(open_mask)[0]
        m = len(open_idx)
        if m == 0:
            return None, None

        # Variables: x[c, k] for customer c, open facility k (0-indexed in open set)
        # n_vars = n_c * m
        n_vars = n_c * m

        # Objective coefficients
        c_obj = assign_cost[:, open_idx].ravel().copy()

        # Equality constraints: each customer assigned to exactly one facility
        # A_eq: n_c x n_vars
        A_eq = np.zeros((n_c, n_vars))
        for c in range(n_c):
            A_eq[c, c * m:(c + 1) * m] = 1.0
        b_eq = np.ones(n_c)

        # Inequality constraints: capacity of each open facility
        # A_ub: m x n_vars
        A_ub = np.zeros((m, n_vars))
        for k in range(m):
            for c in range(n_c):
                A_ub[k, c * m + k] = 1.0
        b_ub = capacities[open_idx]

        bounds = [(0, 1)] * n_vars

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
        except Exception:
            return None, None

        if res.status != 0:
            return None, None

        if time.time() - t0 > max_time:
            return None, None

        x = res.x.reshape(n_c, m)

        # The LP solution should be integral for this network structure.
        # If not, round carefully: for each customer, pick the facility with highest x.
        assignment = np.zeros(n_c, dtype=int)
        for c in range(n_c):
            k_best = int(np.argmax(x[c]))
            assignment[c] = open_idx[k_best]

        # Verify feasibility
        load = np.zeros(n_f)
        for c in range(n_c):
            load[assignment[c]] += demands[c]
        if np.any(load > capacities + 1e-9):
            # Rounding broke feasibility; try a greedy repair
            return None, None

        return assignment, res.fun

    def greedy_initial():
        """Greedy assignment: sort customers by demand desc, assign to cheapest feasible facility."""
        assignment = np.zeros(n_c, dtype=int)
        load = np.zeros(n_f)
        order = np.argsort(-demands)
        for c in order:
            feasible = load + demands[c] <= capacities + 1e-9
            if not feasible.any():
                feasible = np.ones(n_f, dtype=bool)
            costs = assign_cost[c].copy()
            costs[~feasible] = np.inf
            f = int(np.argmin(costs))
            assignment[c] = f
            load[f] += demands[c]
        return assignment

    def local_search(assignment, open_mask, max_passes=50):
        """
        Local search: repeatedly try moving a single customer to a different open facility
        if it reduces total cost.
        """
        assignment = assignment.copy()
        load = np.zeros(n_f)
        for c in range(n_c):
            load[assignment[c]] += demands[c]

        current_cost = total_cost(assignment)
        if np.isinf(current_cost):
            return assignment, current_cost

        open_set = np.where(open_mask)[0]

        for _ in range(max_passes):
            if time.time() > deadline - 5:
                break
            improved = False
            for c in range(n_c):
                old_f = int(assignment[c])
                if not open_mask[old_f]:
                    # Shouldn't happen, but fix it
                    feasible = (load + demands[c] <= capacities + 1e-9) & open_mask
                    if feasible.any():
                        new_f = int(np.argmin(assign_cost[c][feasible]))
                        load[old_f] -= demands[c]
                        assignment[c] = new_f
                        load[new_f] += demands[c]
                        current_cost = total_cost(assignment)
                        continue
                    continue

                load[old_f] -= demands[c]

                best_f = -1
                best_delta = 0.0  # Only accept improvements (delta < 0)

                for new_f in open_set:
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f] + 1e-9:
                        continue

                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]

                    # Open cost changes
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]
                    if load[new_f] == 0:
                        delta += open_cost[new_f]

                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_f = new_f

                if best_f != -1 and best_delta < -1e-9:
                    assignment[c] = best_f
                    load[best_f] += demands[c]
                    current_cost += best_delta
                    improved = True
                else:
                    load[old_f] += demands[c]

            if not improved:
                break

        return assignment, current_cost

    def two_opt_swap(assignment, open_mask, max_passes=10):
        """
        Try swapping two customers' assignments if it reduces cost.
        Only consider swaps between open facilities.
        """
        assignment = assignment.copy()
        load = np.zeros(n_f)
        for c in range(n_c):
            load[assignment[c]] += demands[c]

        current_cost = total_cost(assignment)
        if np.isinf(current_cost):
            return assignment, current_cost

        open_set = np.where(open_mask)[0]

        for _ in range(max_passes):
            if time.time() > deadline - 10:
                break
            improved = False
            # Sample pairs to keep it fast
            for _ in range(200):
                c1 = np.random.randint(n_c)
                c2 = np.random.randint(n_c)
                if c1 == c2:
                    continue

                f1 = int(assignment[c1])
                f2 = int(assignment[c2])
                if f1 == f2:
                    continue
                if not open_mask[f1] or not open_mask[f2]:
                    continue

                # Check capacity feasibility after swap
                # Customer c1 moves to f2, customer c2 moves to f1
                # load[f1] changes by: -demands[c1] + demands[c2]
                # load[f2] changes by: -demands[c2] + demands[c1]
                new_load_f1 = load[f1] - demands[c1] + demands[c2]
                new_load_f2 = load[f2] - demands[c2] + demands[c1]

                if new_load_f1 > capacities[f1] + 1e-9 or new_load_f2 > capacities[f2] + 1e-9:
                    continue
                if new_load_f1 < -1e-9 or new_load_f2 < -1e-9:
                    continue

                delta_assign = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                               (assign_cost[c1, f1] + assign_cost[c2, f2])

                delta_open = 0.0
                if load[f1] == demands[c1]:  # f1 becomes empty
                    delta_open -= open_cost[f1]
                if load[f2] == demands[c2]:  # f2 becomes empty
                    delta_open -= open_cost[f2]
                if load[f1] == 0:  # f1 was empty (but c1 is there, so load[f1]>=demands[c1]>0)
                    pass
                if load[f2] == 0:
                    pass

                total_delta = delta_assign + delta_open
                if total_delta < -1e-9:
                    # Perform swap
                    assignment[c1] = f2
                    assignment[c2] = f1
                    load[f1] += demands[c2] - demands[c1]
                    load[f2] += demands[c1] - demands[c2]
                    current_cost += total_delta
                    improved = True
                    break  # Restart scan after improvement

            if not improved:
                break

        return assignment, current_cost

    # --- Main Strategy ---

    # 1. Greedy initial solution
    assignment = greedy_initial()
    current_cost = total_cost(assignment)
    if np.isinf(current_cost):
        # Fallback: assign everyone to facility 0 (shouldn't happen with valid input)
        assignment = np.zeros(n_c, dtype=int)
        current_cost = total_cost(assignment)

    # 2. Determine initial open set: all facilities that have at least one customer
    open_mask = np.zeros(n_f, dtype=bool)
    for c in range(n_c):
        open_mask[int(assignment[c])] = True

    # 3. Solve min-cost flow on the initial open set to get a better assignment
    if time.time() < deadline - 30:
        new_assign, flow_cost = solve_flow(open_mask, max_time=15.0)
        if new_assign is not None:
            new_total = total_cost(new_assign)
            if new_total < current_cost:
                assignment = new_assign
                current_cost = new_total
                # Update open mask
                open_mask = np.zeros(n_f, dtype=bool)
                for c in range(n_c):
                    open_mask[int(assignment[c])] = True

    # 4. Iteratively try to close facilities
    # Compute load
    load = np.zeros(n_f)
    for c in range(n_c):
        load[assignment[c]] += demands[c]

    # Try closing each open facility one at a time (in order of high open cost / low usage)
    # Priority: close facilities with high open_cost and low load
    open_facilities = np.where(open_mask)[0]
    # Sort by open_cost / max(load, 1) descending — close expensive, lightly used ones first
    if len(open_facilities) > 0:
        usage_ratio = np.zeros(len(open_facilities))
        for i, f in enumerate(open_facilities):
            usage_ratio[i] = open_cost[f] / max(load[f], 1.0)
        close_order = np.argsort(-usage_ratio)
    else:
        close_order = []

    for idx in close_order:
        if time.time() > deadline - 40:
            break
        f = int(open_facilities[idx])
        if not open_mask[f]:
            continue

        new_open = open_mask.copy()
        new_open[f] = False

        # Check if remaining capacity is sufficient
        if capacities[new_open].sum() < demands.sum() + 1e-9:
            continue

        new_assign, flow_cost = solve_flow(new_open, max_time=8.0)
        if new_assign is None:
            continue

        new_cost = total_cost(new_assign)
        if new_cost < current_cost - 1e-9:
            current_cost = new_cost
            assignment = new_assign
            open_mask = new_open
            # Recompute load
            load = np.zeros(n_f)
            for c in range(n_c):
                load[assignment[c]] += demands[c]
            # Update open_facilities list (remove f)
            open_facilities = open_facilities[np.where(open_facilities != f)[0]]
            # Re-sort for next iteration? Keep it simple: just continue with remaining list
            # Actually, let's rebuild
            open_facilities = np.where(open_mask)[0]
            if len(open_facilities) > 0:
                usage_ratio = np.zeros(len(open_facilities))
                for i, f2 in enumerate(open_facilities):
                    usage_ratio[i] = open_cost[f2] / max(load[f2], 1.0)
                close_order = np.argsort(-usage_ratio)
            else:
                close_order = []
            break  # Restart the loop with updated state

    # 5. Local search on the current assignment
    if time.time() < deadline - 15:
        assignment, current_cost = local_search(assignment, open_mask, max_passes=30)

    # 6. Two-opt swap
    if time.time() < deadline - 10:
        assignment, current_cost = two_opt_swap(assignment, open_mask, max_passes=5)

    # 7. Final local search pass
    if time.time() < deadline - 5:
        assignment, current_cost = local_search(assignment, open_mask, max_passes=10)

    return assignment.astype(int)

# EVOLVE-BLOCK-END

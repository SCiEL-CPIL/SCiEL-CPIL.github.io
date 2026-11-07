import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_customers = len(demands)
    n_facilities = len(capacities)

    def calc_cost(assignment, load):
        total = 0.0
        for j in range(n_facilities):
            if load[j] > 0:
                total += open_cost[j]
        for i in range(n_customers):
            j = assignment[i]
            if j >= 0:
                total += assign_cost[i, j]
        return total

    def valid_assignment(assignment, load):
        if np.any(load < -1e-9):
            return False
        if np.any(load > capacities + 1e-9):
            return False
        if np.any(assignment < 0) or np.any(assignment >= n_facilities):
            return False
        return True

    def lp_guided_init():
        # LP relaxation for CFLP
        # Variables: x[i,j] for assignment (n_customers * n_facilities)
        #            y[j] for opening (n_facilities)
        # Minimize: sum(assign_cost[i,j] * x[i,j]) + sum(open_cost[j] * y[j])
        # Subject to:
        #   sum_j x[i,j] = 1 for all i (each customer assigned to one facility)
        #   sum_i demands[i] * x[i,j] <= capacities[j] for all j (capacity)
        #   x[i,j] <= y[j] for all i,j (linking)
        #   0 <= x[i,j] <= 1, 0 <= y[j] <= 1

        n_vars = n_customers * n_facilities + n_facilities

        # Objective
        c_x = assign_cost.flatten()
        c_y = open_cost
        c = np.concatenate([c_x, c_y])

        # Equality constraints: sum_j x[i,j] = 1 for each customer
        A_eq = lil_matrix((n_customers, n_vars))
        b_eq = np.ones(n_customers)
        for i in range(n_customers):
            for j in range(n_facilities):
                A_eq[i, i * n_facilities + j] = 1.0

        # Inequality constraints:
        # Capacity: sum_i demands[i] * x[i,j] <= capacities[j] for each j
        # Linking: x[i,j] - y[j] <= 0 for each i,j
        n_ineq = n_facilities + n_customers * n_facilities
        A_ub = lil_matrix((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)

        # Capacity constraints (first n_facilities rows)
        for j in range(n_facilities):
            for i in range(n_customers):
                A_ub[j, i * n_facilities + j] = demands[i]
            b_ub[j] = capacities[j]

        # Linking constraints (remaining rows)
        row = n_facilities
        for i in range(n_customers):
            for j in range(n_facilities):
                A_ub[row, i * n_facilities + j] = 1.0
                A_ub[row, n_customers * n_facilities + j] = -1.0
                row += 1

        bounds = [(0, 1)] * (n_customers * n_facilities) + [(0, 1)] * n_facilities

        try:
            res = linprog(c, A_ub=A_ub.tocsr(), b_ub=b_ub, A_eq=A_eq.tocsr(), b_eq=b_eq,
                          bounds=bounds, method='highs')
            if res.status == 0:
                x_frac = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
                # Greedy rounding based on LP solution
                assignment = np.full(n_customers, -1, dtype=int)
                load = np.zeros(n_facilities, dtype=np.float64)
                # Sort customers by demand descending
                cust_order = np.argsort(-demands)
                for i in cust_order:
                    d_i = demands[i]
                    best_j = -1
                    best_score = float('inf')
                    for j in range(n_facilities):
                        if load[j] + d_i <= capacities[j]:
                            score = assign_cost[i, j] - 0.5 * open_cost[j] * (1.0 if load[j] == 0 else 0.0)
                            # Use LP fractional value as tiebreaker
                            score -= 100 * x_frac[i, j]
                            if score < best_score:
                                best_score = score
                                best_j = j
                    if best_j == -1:
                        # Fallback
                        best_j = np.argmax(capacities - load)
                    assignment[i] = best_j
                    load[best_j] += d_i
                return assignment, load
        except Exception:
            pass

        # Fallback: pure greedy
        assignment = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities, dtype=np.float64)
        cust_order = np.argsort(-demands)
        for i in cust_order:
            d_i = demands[i]
            best_j = -1
            best_cost = float('inf')
            for j in range(n_facilities):
                if load[j] + d_i <= capacities[j]:
                    mc = assign_cost[i, j]
                    if load[j] == 0:
                        mc += open_cost[j]
                    if mc < best_cost:
                        best_cost = mc
                        best_j = j
            if best_j == -1:
                best_j = np.argmax(capacities - load)
            assignment[i] = best_j
            load[best_j] += d_i
        return assignment, load

    # Initial solution
    assignment, load = lp_guided_init()
    if not valid_assignment(assignment, load):
        # Reset to simple greedy
        assignment = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities, dtype=np.float64)
        cust_order = np.argsort(-demands)
        for i in cust_order:
            d_i = demands[i]
            best_j = -1
            best_cost = float('inf')
            for j in range(n_facilities):
                if load[j] + d_i <= capacities[j]:
                    mc = assign_cost[i, j]
                    if load[j] == 0:
                        mc += open_cost[j]
                    if mc < best_cost:
                        best_cost = mc
                        best_j = j
            if best_j == -1:
                best_j = 0
            assignment[i] = best_j
            load[best_j] += d_i

    current_cost = calc_cost(assignment, load)

    # Local search with 1-moves and 2-swaps
    no_improve = 0
    max_no_improve = 30

    while time.time() < deadline and no_improve < max_no_improve:
        improved = False

        # 1-moves: move customer i from j to k
        for i in range(n_customers):
            if time.time() > deadline:
                break
            j = assignment[i]
            d_i = demands[i]
            best_k = -1
            best_delta = -1e-10
            for k in range(n_facilities):
                if k == j:
                    continue
                if load[k] + d_i <= capacities[k]:
                    delta = assign_cost[i, k] - assign_cost[i, j]
                    if load[j] - d_i == 0:
                        delta -= open_cost[j]
                    if load[k] == 0:
                        delta += open_cost[k]
                    if delta < best_delta:
                        best_delta = delta
                        best_k = k
            if best_k != -1:
                load[j] -= d_i
                load[best_k] += d_i
                assignment[i] = best_k
                current_cost += best_delta
                improved = True

        if not improved:
            # 2-swaps: swap customers i and l
            found_swap = False
            for i in range(n_customers):
                if found_swap:
                    break
                j = assignment[i]
                d_i = demands[i]
                best_l = -1
                best_delta = -1e-10
                for l in range(i + 1, n_customers):
                    k = assignment[l]
                    if j == k:
                        continue
                    d_l = demands[l]
                    new_load_j = load[j] - d_i + d_l
                    new_load_k = load[k] - d_l + d_i
                    if new_load_j <= capacities[j] and new_load_k <= capacities[k]:
                        delta_assign = (assign_cost[i, k] - assign_cost[i, j]) + (assign_cost[l, j] - assign_cost[l, k])
                        delta_open = 0.0
                        if new_load_j == 0:
                            delta_open -= open_cost[j]
                        if new_load_k == 0:
                            delta_open -= open_cost[k]
                        total_delta = delta_assign + delta_open
                        if total_delta < best_delta:
                            best_delta = total_delta
                            best_l = l
                if best_l != -1:
                    l = best_l
                    k = assignment[l]
                    d_l = demands[l]
                    load[j] -= d_i
                    load[k] -= d_l
                    load[j] += d_l
                    load[k] += d_i
                    assignment[i] = k
                    assignment[l] = j
                    current_cost += best_delta
                    improved = True
                    found_swap = True

            if found_swap:
                no_improve = 0
                continue

        if improved:
            no_improve = 0
        else:
            no_improve += 1

    return assignment

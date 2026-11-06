import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 115  # Safety margin (evaluator kills at 150s)

    def is_valid(A):
        """Check if the graph has no K2,3 subgraph."""
        K = A @ A
        np.fill_diagonal(K, 0)
        return np.max(K) <= 2

    def count_edges(A):
        return int(np.sum(A)) // 2

    def try_add_edge(A, u, v):
        """Try to add edge (u, v). Returns True if added, False if not."""
        if A[u, v] == 1:
            return False
        for w in range(n):
            if w == u or w == v:
                continue
            # Check pair (u, w)
            cn_uw = int(np.dot(A[u, :], A[:, w]))
            if A[v, w] == 1:
                if cn_uw >= 2:
                    return False
            # Check pair (v, w)
            cn_vw = int(np.dot(A[v, :], A[:, w]))
            if A[u, w] == 1:
                if cn_vw >= 2:
                    return False
        A[u, v] = 1
        A[v, u] = 1
        return True

    def greedy_fill(A):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        if try_add_edge(A, u, v):
                            changed = True
        return A

    def local_search(A, budget):
        best_A = A.copy()
        best_edges = count_edges(A)

        while time.time() < budget:
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 0]
            if not non_edges:
                break

            u, v = random.choice(non_edges)
            if try_add_edge(A, u, v):
                current_edges = count_edges(A)
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()
            else:
                edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 1]
                if not edges:
                    break
                ru, rv = random.choice(edges)

                A_backup = A.copy()

                A[ru, rv] = 0
                A[rv, ru] = 0

                try_add_edge(A, u, v)

                non_edges_now = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]
                for _ in range(5):
                    if not non_edges_now:
                        break
                    x, y = random.choice(non_edges_now)
                    if try_add_edge(A, x, y):
                        non_edges_now.remove((x, y))

                current_edges = count_edges(A)

                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()
                elif current_edges < best_edges:
                    A[:] = A_backup

        return best_A, best_edges

    def warm_start_greedy():
        """Build a valid K2,3-free graph via 5 K4s + greedy fill, as MILP warm start."""
        A = np.zeros((n, n), dtype=int)
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    A[u, v] = 1
                    A[v, u] = 1
        greedy_fill(A)
        return A

    def solve_milp(warm_A, time_limit):
        """Solve the linearized MILP for the densest K2,3-free graph.
        Variables: x_{uv} (edge, u<v) and y_{u,v,w} (= x_{uw}*x_{vw}).
        Constraints: y <= x_{uw}, y <= x_{vw}, y >= x_{uw}+x_{vw}-1,
                     sum_w y_{u,v,w} <= 2 for each pair (u,v).
        Returns adjacency matrix (may be fractional -> rounded to a valid graph).
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except Exception:
            return None

        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        idx = {(u, v): i for i, (u, v) in enumerate(pairs)}
        num_x = len(pairs)  # 190

        # Auxiliary y variables: for each pair (u,v) and each w not in {u,v}
        y_info = []  # list of (y_index, u, v, w, x_uw_index, x_vw_index)
        y_index = 0
        for (u, v) in pairs:
            for w in range(n):
                if w == u or w == v:
                    continue
                y_info.append((y_index, u, v, w, idx[(min(u, w), max(u, w))], idx[(min(v, w), max(v, w))]))
                y_index += 1
        num_y = y_index
        total_vars = num_x + num_y

        # Objective: maximize sum x  ->  minimize -sum x
        c = np.zeros(total_vars)
        c[:num_x] = -1.0

        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)

        # Build constraint matrix A_ub (inequalities)
        # We'll accumulate rows: each row is (lo, hi, row_vector)
        rows_lo = []
        rows_hi = []
        rows_vec = []

        def add_row(lo, hi, vec):
            rows_lo.append(lo)
            rows_hi.append(hi)
            rows_vec.append(vec)

        # y <= x_uw  =>  y - x_uw <= 0
        # y <= x_vw  =>  y - x_vw <= 0
        # y >= x_uw + x_vw - 1  =>  -y + x_uw + x_vw <= 1
        # sum_w y_{u,v,w} <= 2  =>  sum y <= 2
        for (yi, u, v, w, i_uw, i_vw) in y_info:
            yv = num_x + yi
            vec = np.zeros(total_vars)
            vec[yv] = 1.0
            vec[i_uw] = -1.0
            add_row(-np.inf, 0.0, vec)

            vec = np.zeros(total_vars)
            vec[yv] = 1.0
            vec[i_vw] = -1.0
            add_row(-np.inf, 0.0, vec)

            vec = np.zeros(total_vars)
            vec[yv] = -1.0
            vec[i_uw] = 1.0
            vec[i_vw] = 1.0
            add_row(-np.inf, 1.0, vec)

        # sum_w y_{u,v,w} <= 2 for each pair (u,v)
        for (u, v) in pairs:
            vec = np.zeros(total_vars)
            for (yi, uu, vv, w, i_uw, i_vw) in y_info:
                if uu == u and vv == v:
                    vec[num_x + yi] = 1.0
            add_row(-np.inf, 2.0, vec)

        A_ub = np.array(rows_vec, dtype=float)
        lb_arr = np.array(rows_lo, dtype=float)
        ub_arr = np.array(rows_hi, dtype=float)

        constraints = LinearConstraint(A_ub, lb_arr, ub_arr)
        bounds = Bounds(lb, ub)

        # Warm start: x from warm_A, y = x_uw * x_vw
        x0 = np.zeros(total_vars)
        for (u, v) in pairs:
            x0[idx[(u, v)]] = warm_A[u, v]
        for (yi, u, v, w, i_uw, i_vw) in y_info:
            x0[num_x + yi] = warm_A[u, w] * warm_A[v, w]

        try:
            res = milp(c=c, constraints=constraints, bounds=bounds,
                       integrality=np.ones(total_vars), x0=x0,
                       options={"time_limit": time_limit, "mip_gap": 0.0,
                                "threads": 4, "presolve": "default"})
        except Exception:
            return None

        if res.x is None:
            return None

        x_sol = res.x[:num_x]
        # Round to nearest integer and rebuild adjacency
        A = np.zeros((n, n), dtype=int)
        for (u, v) in pairs:
            val = int(round(x_sol[idx[(u, v)]]))
            if val > 0:
                A[u, v] = 1
                A[v, u] = 1
        return A

    def heuristic_seed():
        A = np.zeros((n, n), dtype=int)
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    A[u, v] = 1
                    A[v, u] = 1
        for _ in range(20):
            u, v = random.sample(range(n), 2)
            try_add_edge(A, u, v)
        greedy_fill(A)
        return A

    best_overall_A = None
    best_overall_edges = 0

    def consider(A):
        nonlocal best_overall_A, best_overall_edges
        if A is None:
            return
        if not is_valid(A):
            return
        e = count_edges(A)
        if e > best_overall_edges:
            best_overall_edges = e
            best_overall_A = A.copy()

    # Phase 1: Global MILP solve (structural escape from heuristic local optima)
    warm = warm_start_greedy()
    consider(warm)
    milp_budget = min(40, max(5, deadline - time.time() - 25))
    if time.time() < deadline - 25:
        milp_A = solve_milp(warm, milp_budget)
        if milp_A is not None:
            # Round may have introduced violations; repair via greedy removal if needed
            if not is_valid(milp_A):
                # Remove edges until valid (should be rare)
                A = milp_A.copy()
                changed = True
                while changed and is_valid(A) is False:
                    changed = False
                    for u in range(n):
                        for v in range(u + 1, n):
                            if A[u, v] == 1:
                                A[u, v] = 0
                                A[v, u] = 0
                                if is_valid(A):
                                    changed = True
                                    break
                        if changed:
                            break
                milp_A = A
            consider(milp_A)
            # Polish the MILP result with greedy + local search
            if is_valid(milp_A):
                milp_A = greedy_fill(milp_A)
                consider(milp_A)
                polish_end = min(time.time() + 15, deadline - 10)
                if time.time() < polish_end:
                    pol_A, pol_e = local_search(milp_A.copy(), polish_end)
                    consider(pol_A)

    # Phase 2: Heuristic restarts with greedy + local search (hedge + refinement)
    while time.time() < deadline - 5:
        A = heuristic_seed()
        consider(A)
        slice_end = min(time.time() + 6, deadline - 5)
        if time.time() < slice_end:
            A, e = local_search(A, slice_end)
            consider(A)

    # Final polish on best solution
    if best_overall_A is not None and time.time() < deadline - 3:
        final_end = min(time.time() + 8, deadline - 2)
        fin_A, fin_e = local_search(best_overall_A.copy(), final_end)
        consider(fin_A)

    # Convert to edge list
    edges_list = []
    if best_overall_A is not None and is_valid(best_overall_A):
        for u in range(n):
            for v in range(u + 1, n):
                if best_overall_A[u, v]:
                    edges_list.append([u, v])
    else:
        # Fallback: 5 disjoint K4s
        A = np.zeros((n, n), dtype=int)
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    A[u, v] = 1
                    A[v, u] = 1
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v]:
                    edges_list.append([u, v])

    return edges_list

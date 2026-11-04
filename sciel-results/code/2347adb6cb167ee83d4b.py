# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    # 1. Initial Heuristic: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    # High flow facilities at low distance locations
    fac_order = np.argsort(-a_row_sums, kind="stable")
    loc_order = np.argsort(b_row_sums, kind="stable")

    p_init = np.empty(n, dtype=np.int64)
    p_init[fac_order] = loc_order

    current_cost = compute_cost(p_init)
    best_cost = current_cost
    best_p = p_init.copy()

    # 2. Continuous Relaxation: FISTA on Birkhoff Polytope
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0

    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p_init] = 1.0

    alpha = 0.2
    t_k = 1.0
    X_prev = X.copy()

    # Run FISTA for a limited time
    fista_deadline = start_time + 30.0  # Reserve 30s for FISTA
    for k in range(2000):
        if time.time() > fista_deadline:
            break

        # Gradient
        AX = A_sym @ X
        G = AX @ B_sym + B_sym @ AX

        # FISTA Update
        t_prev = t_k
        t_k = (1 + np.sqrt(1 + 4 * t_prev * t_prev)) / 2
        Y = X + (t_prev - 1) / t_k * (X - X_prev)

        # Gradient Step
        X_new = Y - alpha * G

        # Project onto Birkhoff Polytope
        row_ind, col_ind = linear_sum_assignment(-X_new)
        X_proj = np.zeros((n, n), dtype=np.float64)
        X_proj[row_ind, col_ind] = 1.0

        p_relax = np.empty(n, dtype=np.int64)
        p_relax[row_ind] = col_ind
        cost_relax = compute_cost(p_relax)

        if cost_relax < best_cost:
            best_cost = cost_relax
            best_p = p_relax.copy()

        X_prev = X
        X = X_proj

    # 3. Local Search: Vectorized 2-Swap
    p = best_p.copy()
    current_cost = best_cost

    # Precompute b_p = b[p, :] and b_pT = b[:, p] for faster access
    b_p = b[p, :]  # Shape (n, n): b_p[i, j] = b[p[i], j]
    b_pT = b[:, p] # Shape (n, n): b_pT[i, j] = b[i, p[j]]

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1

        for i in range(n):
            if time.time() > deadline - 5.0:
                break

            u = p[i]
            A_i = a[i, :]
            A_iT = a[:, i]

            # Precompute differences for this i
            # diff1[k] = b[v, p[k]] - b[u, p[k]] where v = p[j]
            # diff2[k] = b[p[k], v] - b[p[k], u]

            for j in range(i + 1, n):
                v = p[j]
                A_j = a[j, :]
                A_jT = a[:, j]

                D1 = A_i - A_j
                D2 = A_iT - A_jT

                # Vectorized sums
                diff1 = b[v, p] - b[u, p]
                diff2 = b[p, v] - b[p, u]

                Sum1_all = D1 @ diff1
                Sum2_all = D2 @ diff2

                # Subtract k=i and k=j terms
                # k=i:
                Sub1_i = D1[i] * (b[v, u] - b[u, u])
                Sub2_i = D2[i] * (b[u, v] - b[u, u])

                # k=j:
                Sub1_j = D1[j] * (b[v, v] - b[u, v])
                Sub2_j = D2[j] * (b[v, v] - b[v, u])

                Sum1 = Sum1_all - Sub1_i - Sub1_j
                Sum2 = Sum2_all - Sub2_i - Sub2_j

                Cross = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])

                delta = Cross + Sum1 + Sum2

                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        if best_delta < -1e-10:
            # Perform swap
            i, j = best_i, best_j
            p[i], p[j] = p[j], p[i]
            current_cost += best_delta

            # Update b_p and b_pT
            u, v = p[i], p[j]
            b_p[i, :] = b[v, :]
            b_p[j, :] = b[u, :]
            b_pT[:, i] = b[:, v]
            b_pT[:, j] = b[:, u]

            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
        else:
            break

    # 4. Simulated Annealing Phase
    # Use remaining time for SA to escape local minima
    sa_deadline = deadline - 2.0
    if time.time() < sa_deadline:
        p = best_p.copy()
        current_cost = best_cost
        b_p = b[p, :]
        b_pT = b[:, p]

        # Temperature schedule
        T_start = max(1.0, abs(best_cost) * 0.01)
        T_end = 1e-6
        cooling_rate = 0.995

        T = T_start
        rng = random.Random(42)

        while T > T_end and time.time() < sa_deadline:
            # Pick random pair
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue

            u, v = p[i], p[j]

            # Compute delta
            A_i = a[i, :]
            A_iT = a[:, i]
            A_j = a[j, :]
            A_jT = a[:, j]

            D1 = A_i - A_j
            D2 = A_iT - A_jT

            diff1 = b[v, p] - b[u, p]
            diff2 = b[p, v] - b[p, u]

            Sum1_all = D1 @ diff1
            Sum2_all = D2 @ diff2

            Sub1_i = D1[i] * (b[v, u] - b[u, u])
            Sub2_i = D2[i] * (b[u, v] - b[u, u])
            Sub1_j = D1[j] * (b[v, v] - b[u, v])
            Sub2_j = D2[j] * (b[v, v] - b[v, u])

            Sum1 = Sum1_all - Sub1_i - Sub1_j
            Sum2 = Sum2_all - Sub2_i - Sub2_j

            Cross = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
            delta = Cross + Sum1 + Sum2

            # Accept or reject
            if delta < 0 or rng.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta

                # Update b_p and b_pT
                u, v = p[i], p[j]
                b_p[i, :] = b[v, :]
                b_p[j, :] = b[u, :]
                b_pT[:, i] = b[:, v]
                b_pT[:, j] = b[:, u]

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()

            T *= cooling_rate

    return best_p.tolist()
# EVOLVE-BLOCK-END

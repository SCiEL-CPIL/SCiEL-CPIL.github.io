"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    """
    Solve the Quadratic Assignment Problem (QAP) for the given matrices a and b.
    Returns a permutation p of 0..n-1 that minimizes sum_{i,j} a[i,j] * b[p[i], p[j]].
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Ensure we work with int64 arrays for correctness of the exact integer cost
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)

    # ---- Initial heuristic: rank matching (heavy flow -> central/low-distance) ----
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    central_order = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order

    def compute_cost(p):
        b_sub = b[np.ix_(p, p)]
        return int(np.sum(a * b_sub))

    current_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # ---- Fast vectorized 2-opt (swap) delta ----
    # For swapping locations of facilities i and j (li=p[i], lj=p[j]):
    #   delta = dot(a[i]-a[j], b[lj,p]-b[li,p]) + dot(a[:,i]-a[:,j], b[p,lj]-b[p,li])
    # This is O(n) per pair but with tiny constant (two dot products), no masking.
    def swap_delta(p, i, j):
        li = p[i]
        lj = p[j]
        d1 = np.dot(a[i] - a[j], b[lj, p] - b[li, p])
        d2 = np.dot(a[:, i] - a[:, j], b[p, lj] - b[p, li])
        return d1 + d2

    # ---- Steepest-descent 2-opt local search with time guard ----
    improved = True
    stall_rounds = 0
    while improved and (time.time() - start_time) < deadline - 5.0:
        improved = False
        best_delta = 0
        best_pair = (-1, -1)

        # Full pass over all pairs
        for i in range(n):
            if (time.time() - start_time) > deadline - 5.0:
                break
            ai = a[i]
            ai_col = a[:, i]
            for j in range(i + 1, n):
                d = swap_delta(p, i, j)
                if d < best_delta:
                    best_delta = d
                    best_pair = (i, j)

        if best_pair[0] != -1:
            i, j = best_pair
            p[i], p[j] = p[j], p[i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
            stall_rounds = 0
        else:
            stall_rounds += 1

    # ---- Random restarts with perturbation + local search if time remains ----
    rng = np.random.default_rng(42)
    while (time.time() - start_time) < deadline - 10.0:
        # Perturb the best solution
        p_restart = best_p.copy()
        n_swaps = max(5, n // 15)
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p_restart[i], p_restart[j] = p_restart[j], p_restart[i]

        current_cost_r = compute_cost(p_restart)

        # Local search from perturbed solution (with time guard)
        improved_r = True
        while improved_r and (time.time() - start_time) < deadline - 5.0:
            improved_r = False
            best_delta_r = 0
            best_pair_r = (-1, -1)

            for i in range(n):
                if (time.time() - start_time) > deadline - 5.0:
                    break
                for j in range(i + 1, n):
                    d = swap_delta(p_restart, i, j)
                    if d < best_delta_r:
                        best_delta_r = d
                        best_pair_r = (i, j)

            if best_pair_r[0] != -1:
                i, j = best_pair_r
                p_restart[i], p_restart[j] = p_restart[j], p_restart[i]
                current_cost_r += best_delta_r
                if current_cost_r < best_cost:
                    best_cost = current_cost_r
                    best_p = p_restart.copy()
                improved_r = True

    return best_p.tolist()
# EVOLVE-BLOCK-END

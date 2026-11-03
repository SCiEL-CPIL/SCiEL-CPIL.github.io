# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Compute exact delta for swapping facilities at positions i and j."""
    pi = p[i]
    pj = p[j]
    dB = b[pj, p] - b[pi, p]
    dA = a[i, :] - a[:, j]
    return int(np.dot(dA, dB))


def _hungarian_initial(a, b):
    """
    Initialize via Hungarian algorithm on a surrogate cost matrix.
    The surrogate cost for assigning facility i to location j is based on
    the interaction of facility i's flow pattern with location j's distance pattern.
    
    We use: cost[i, j] = sum_k a[i, k] * b[j, k]
    This is the "row-wise" contribution. Minimizing this gives a good starting point.
    """
    n = a.shape[0]
    # cost_matrix[i, j] = sum_k a[i,k] * b[j,k] = (a @ b.T)[i, j]
    cost_matrix = a.astype(np.float64) @ b.astype(np.float64).T
    # Hungarian minimizes total cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _local_search(a, b, p, deadline, best_cost, best_p):
    """
    First-improvement 2-opt local search with a multi-pass strategy.
    Uses a vectorized approach for delta computation where possible.
    """
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    # Precompute for faster delta: 
    # delta(i,j) = sum_k (a[i,k] - a[k,j]) * (b[p[j],p[k]] - b[p[i],p[k]])
    # We'll iterate but with early termination on time.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            pi = p[i]
            # Precompute b[pi, p] for this i
            b_pi_p = b[pi, p]
            a_i = a[i, :]
            for j in range(i + 1, n):
                pj = p[j]
                a_j_col = a[:, j]
                # delta = sum_k (a[i,k] - a[k,j]) * (b[p[j],p[k]] - b[p[i],p[k]])
                # = sum_k (a[i,k] - a[k,j]) * (b[pj,p[k]] - b[pi,p[k]])
                dB = b[pj, p] - b_pi_p
                dA = a_i - a_j_col
                delta = int(np.dot(dA, dB))
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += delta
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    if time.time() >= deadline:
                        return best_cost, best_p
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    return best_cost, best_p


def _anneal(a, b, p, start_time, deadline, best_cost, best_p, rng):
    """
    Simulated Annealing with adaptive cooling and restarts if stuck.
    """
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    T = 10000.0
    T_end = 1.0
    alpha = 0.998
    no_improve = 0
    max_no_improve = 2000

    while T > T_end and (time.time() - start_time) < deadline:
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue

        delta = _swap_delta(a, b, p, i, j)

        if delta < 0 or (rng.random() < np.exp(-delta / T)):
            p[i], p[j] = p[j], p[i]
            cur += delta

            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
                no_improve = 0
            else:
                no_improve += 1

        T *= alpha

        if no_improve > max_no_improve:
            # Restart with a lower temperature
            T = max(T * 0.5, T_end * 2)
            no_improve = 0

        if time.time() - start_time > deadline:
            break

    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    return best_cost, best_p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s kill

    # --- Phase 1: Hungarian-based initialization ---
    p0 = _hungarian_initial(a, b)
    best_cost = _qap_cost(a, b, p0)
    best_p = p0.copy()

    # --- Phase 2: Local search from Hungarian start ---
    p = p0.copy()
    best_cost, best_p = _local_search(a, b, p, deadline, best_cost, best_p)

    # --- Phase 3: Simulated Annealing for diversification ---
    rng = np.random.default_rng(42)
    p = best_p.copy()
    best_cost, best_p = _anneal(a, b, p, t0, deadline, best_cost, best_p, rng)

    # --- Phase 4: Multiple random restarts with local search ---
    while time.time() < deadline - 5.0:
        p = rng.permutation(n).astype(np.int64)
        # Quick local search
        best_cost, best_p = _local_search(a, b, p, deadline, best_cost, best_p)

    return best_p.tolist()
# EVOLVE-BLOCK-END

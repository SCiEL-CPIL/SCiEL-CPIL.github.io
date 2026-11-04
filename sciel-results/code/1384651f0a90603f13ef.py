"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(p, a, b):
    """Exact integer QAP cost for permutation p."""
    # p[i] = location of facility i
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Vectorized: build index matrix
    idx = np.ix_(p, p)
    return int(np.sum(a * b[idx]))


def _lin_cost_matrix(p, a, b):
    """
    Build the linearized cost matrix for the assignment problem.
    C[i, j] = sum_k a[i,k] * b[j, p[k]]
    This is the cost of placing facility i at location j, given the
    current assignment p for the other facilities.
    """
    # b_p = b with rows permuted by p: b_p[j, k] = b[j, p[k]]
    b_p = b[:, p]
    # C[i, j] = sum_k a[i,k] * b_p[j, k] = (a @ b_p.T)[i, j]
    return a @ b_p.T


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    # Initial heuristic: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    best_p = p.copy()
    best_cost = _cost(p, a, b)

    # Iterative Linear Assignment (ILAP) / block-coordinate descent
    # Alternate between:
    #   1. Fix locations, reassign facilities (solve assignment)
    #   2. Fix facilities, reassign locations (solve assignment)
    # In practice for QAP, we can do:
    #   - Build linearized cost matrix C where C[i,j] = sum_k a[i,k]*b[j,p[k]]
    #   - Solve min-sum assignment on C
    #   - Repeat until no improvement or time runs out

    consecutive_no_improve = 0
    max_no_improve = 10

    while time.time() < deadline:
        # Build linearized cost matrix
        C = _lin_cost_matrix(p, a, b)

        # Solve the linear assignment problem
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind

        # Evaluate
        new_cost = _cost(new_p, a, b)

        if new_cost < best_cost:
            best_cost = new_cost
            best_p = new_p.copy()
            p = new_p.copy()
            consecutive_no_improve = 0
        else:
            # Try a perturbation to escape local optima
            consecutive_no_improve += 1
            if consecutive_no_improve >= max_no_improve:
                # Perturb: swap two random facilities
                if n >= 2:
                    i, j = np.random.choice(n, size=2, replace=False)
                    new_p2 = p.copy()
                    new_p2[i], new_p2[j] = new_p2[j], new_p2[i]
                    p = new_p2
                consecutive_no_improve = 0
                max_no_improve = max(5, max_no_improve // 2)
            else:
                # Try small random swaps
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    i, j = np.random.choice(n, size=2, replace=False)
                    p[i], p[j] = p[j], p[i]
                    c = _cost(p, a, b)
                    if c < best_cost:
                        best_cost = c
                        best_p = p.copy()
                        break
                    else:
                        p[i], p[j] = p[j], p[i]  # revert

    # Final polish: try 2-opt style swaps on the best solution
    if time.time() < deadline - 5:
        p = best_p.copy()
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        break
                    p[i], p[j] = p[j], p[i]
                    c = _cost(p, a, b)
                    if c < best_cost:
                        best_cost = c
                        best_p = p.copy()
                        improved = True
                    else:
                        p[i], p[j] = p[j], p[i]  # revert

    return best_p.tolist()
# EVOLVE-BLOCK-END

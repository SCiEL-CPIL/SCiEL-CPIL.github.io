"""Matrix multiplication 3x5x7 - rank minimization."""
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Starts with schoolbook (rank 105) and attempts to find lower rank decompositions
    via structured bilinear search with half-integer polishing.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n  # 15, 35, 21

    # Schoolbook rank
    rank = n * m * p  # 105

    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    def _verify_np(U_arr, V_arr, W_arr, n, m, p, tol=1e-6):
        """Fast verification using numpy."""
        U_np = np.asarray(U_arr, dtype=float)
        V_np = np.asarray(V_arr, dtype=float)
        W_np = np.asarray(W_arr, dtype=float)
        r = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def _half_int(x):
        return int(round(x * 2)) / 2.0

    def _polish(U_list, V_list, W_list, n, m, p, r):
        """Round to half-integers and verify. Returns (U,V,W) if valid else None."""
        U2 = [[_half_int(v) for v in row] for row in U_list]
        V2 = [[_half_int(v) for v in row] for row in V_list]
        W2 = [[_half_int(v) for v in row] for row in W_list]
        if _verify_np(U2, V2, W2, n, m, p):
            return U2, V2, W2
        return None

    def _build_matrix_for_U(V, W, i, n, m, p, r):
        """A_mat for fixed i: rows (j,k), cols (j,c). A[j*p+k, j*r+c] = V[j*p+k,c]*W[k*n+i,c]."""
        A = np.zeros((m * p, m * r))
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    A[j*p + k, j*r + c] = V[j*p+k][c] * W[k*n+i][c]
        return A

    def _solve_for_U(V, W, n, m, p, r):
        """Given V,W, solve linear systems for each i. Returns U or None."""
        U = [[0.0]*r for _ in range(n*m)]
        b = np.ones(m*p)
        for i in range(n):
            A = _build_matrix_for_U(V, W, i, n, m, p, r)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for j in range(m):
                for c in range(r):
                    U[i*m + j][c] = sol[j*r + c]
        return U

    def _solve_for_W(U, V, n, m, p, r):
        """Given U,V, solve linear systems for each i. Returns W or None."""
        W = [[0.0]*r for _ in range(p*n)]
        for i in range(n):
            # Unknowns: W[k*n+i, c] for k in 0..p-1, c in 0..r-1 -> p*r
            # Eq (j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            A = np.zeros((m*p, p*r))
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        A[j*p + k, k*r + c] = U[i*m+j][c] * V[j*p+k][c]
            b = np.ones(m*p)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for k in range(p):
                for c in range(r):
                    W[k*n + i][c] = sol[k*r + c]
        return W

    def _rand_matrix(rows, r, vals, rng):
        return [[rng.choice(vals) for _ in range(r)] for _ in range(rows)]

    def try_rank(r, seed_base):
        nonlocal best_U, best_V, best_W, best_rank
        if r >= best_rank:
            return
        rng = random.Random(seed_base)
        # candidate value sets
        val_sets = [
            [0, 1],
            [0, 1, 2],
            [-1, 0, 1],
            [0, 1, 2, 3],
        ]
        max_attempts = 100
        for vs_idx, vals in enumerate(val_sets):
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    return
                # random V and W
                V = _rand_matrix(mp, r, vals, rng)
                W = _rand_matrix(pn, r, vals, rng)
                # solve for U
                U = _solve_for_U(V, W, n, m, p, r)
                # polish
                res = _polish(U, V, W, n, m, p, r)
                if res is not None:
                    U2, V2, W2 = res
                    if _verify_np(U2, V2, W2, n, m, p):
                        best_U, best_V, best_W = U2, V2, W2
                        best_rank = r
                        return
                # also try solving for W given U,V (U from above)
                W2 = _solve_for_W(U, V, n, m, p, r)
                res = _polish(U, V, W2, n, m, p, r)
                if res is not None:
                    U3, V3, W3 = res
                    if _verify_np(U3, V3, W3, n, m, p):
                        best_U, best_V, best_W = U3, V3, W3
                        best_rank = r
                        return

    # Search ranks from a reasonable lower bound upward.
    # Lower bound: at least max(n*m, m*p, p*n) = 35, and tensor rank >= that.
    # Try a few candidate ranks below 105.
    for r in range(60, best_rank):
        if time.time() > deadline:
            break
        try_rank(r, 1000 + r)

    return {"U": best_U, "V": best_V, "W": best_W}

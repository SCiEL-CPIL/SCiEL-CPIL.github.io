"""Matrix multiplication 3,3,6 rank-minimal exact decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def verify(U, V, W):
        """Check exact decomposition."""
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if not U or not V or not W:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def snap_and_verify(U, V, W):
        """Snap to half-integers or integers and verify."""
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            if verify(U_list, V_list, W_list):
                return (U_list, V_list, W_list)
        return None

    def try_rank(r, time_budget):
        """Try to find rank-r decomposition via structured random search."""
        end_time = start_time + time_budget
        idxs = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        best_err = float('inf')
        best_U = best_V = best_W = None

        num_restarts = max(1, int((end_time - time.time()) / 0.05))
        num_restarts = min(num_restarts, 200)

        for restart in range(num_restarts):
            if time.time() > end_time:
                break
            # Structured initialization: exploit the 3-block structure
            U = np.random.randn(n * m, r) * 0.3
            V = np.random.randn(m * p, r) * 0.3
            W = np.random.randn(p * n, r) * 0.3

            # Add some structure: initialize with schoolbook-like patterns
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if r > 0:
                            c = (i * m * p + j * p + k) % r
                            U[i * m + j, c] += 0.5
                            V[j * p + k, c] += 0.5
                            W[k * n + i, c] += 0.5

            for step in range(500):
                if time.time() > end_time:
                    break
                # Compute residuals
                errs = np.zeros(n * m * p)
                for ii, (i, j, k) in enumerate(idxs):
                    errs[ii] = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i]) - 1.0
                err = np.sum(errs ** 2)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-12:
                    break
                # Gradient descent with adaptive learning rate
                lr = 0.1 / (1 + step * 0.01)
                for ii, (i, j, k) in enumerate(idxs):
                    e = errs[ii]
                    U[i * m + j] -= lr * e * V[j * p + k] * W[k * n + i]
                    V[j * p + k] -= lr * e * U[i * m + j] * W[k * n + i]
                    W[k * n + i] -= lr * e * U[i * m + j] * V[j * p + k]

            if best_err < 1e-10:
                break

        if best_err < 1e-10:
            result = snap_and_verify(best_U, best_V, best_W)
            if result is not None:
                return result
        return None

    # Try structured decomposition: use rank-23 3x3x3 blocks
    # For each output column i, the sub-problem is 3x3x3 (A row i, B rows 0..5, but
    # actually it's a 3x6 "multiplication" which is 3 copies of 3x3 sharing A)
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But tensor rank lower bound is higher. Try from 18 upward.

    # First try a few specific ranks with more time
    ranks_to_try = list(range(18, 54))

    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        # Allocate more time for lower ranks (harder)
        time_per_rank = min(5.0, (deadline - time.time() - 10) / max(1, len(ranks_to_try) - (r - 18)))
        time_per_rank = max(time_per_rank, 1.0)

        result = try_rank(r, time_per_rank)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END

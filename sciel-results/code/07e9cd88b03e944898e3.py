import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 9
    M = m * p  # 12
    P = p * n  # 12

    # Target tensor T[a, b, c]: T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def timed_out():
        return elapsed() > deadline

    def verify(U, V, W, tol=1e-9):
        # Check half-integer constraint
        if not (np.allclose(U * 2, np.round(U * 2), atol=1e-6) and
                np.allclose(V * 2, np.round(V * 2), atol=1e-6) and
                np.allclose(W * 2, np.round(W * 2), atol=1e-6)):
            return False
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return np.allclose(T_calc, T, atol=tol)

    def snap_to_halfint(U, V, W):
        return (np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2)

    def tensor_error(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return T_calc - T

    # Fast combinatorial repair: search over small half-integer values in {0, ±0.5, ±1, ±1.5, ±2}
    def repair(U, V, W, max_rounds=3):
        vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0])
        U, V, W = snap_to_halfint(U, V, W)
        for _ in range(max_rounds):
            if timed_out():
                break
            err = tensor_error(U, V, W)
            if np.max(np.abs(err)) < 1e-6:
                break
            # Find worst entries
            flat_err = np.abs(err).flatten()
            # Repair the most violated entries by trying value changes on the
            # three tensors at the worst (a,b,c) positions
            worst = np.argsort(flat_err)[::-1][:8]
            improved = False
            for wf in worst:
                if timed_out():
                    break
                a, b, c = np.unravel_index(wf, err.shape)
                # For this (a,b,c), the contribution is sum_r U[a,r]V[b,r]W[c,r]
                # Try changing one factor's value at this index
                # Try U[a, r] for each r
                for r in range(U.shape[1]):
                    for newv in vals:
                        if newv == U[a, r]:
                            continue
                        U[a, r] = newv
                        if np.max(np.abs(tensor_error(U, V, W))) < np.max(np.abs(err)) - 1e-9:
                            err = tensor_error(U, V, W)
                            improved = True
                        else:
                            break
                    # revert if not improved
                    if not improved:
                        pass
                # reset U[a,r] to best (already set)
            if not improved and np.max(np.abs(err)) > 1e-6:
                break
        return U, V, W

    def try_als(R, seed, max_iters=30):
        np.random.seed(seed)
        U = np.random.randn(N, R)
        V = np.random.randn(M, R)
        W = np.random.randn(P, R)

        T_c = T.reshape(N * M, P)
        T_a = T.reshape(N, M * P)
        T_b = T.transpose(1, 0, 2).reshape(M, N * P)

        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        for it in range(max_iters):
            if timed_out():
                break
            # Solve for W
            E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
            W, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
            W = W.T

            # Solve for U
            E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
            U, _, _, _ = np.linalg.lstsq(E_U.T, T_a.T, rcond=None)
            U = U.T

            # Solve for V
            E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
            V, _, _, _ = np.linalg.lstsq(E_V.T, T_b.T, rcond=None)
            V = V.T

            err = np.linalg.norm(tensor_error(U, V, W))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if err < 1e-6:
                break

        # Try repair on the best found
        U_r, V_r, W_r = repair(best_U, best_V, best_W, max_rounds=2)
        if verify(U_r, V_r, W_r):
            return (U_r, V_r, W_r)

        # Also try snapping without repair
        U_s, V_s, W_s = snap_to_halfint(best_U, best_V, best_W)
        if verify(U_s, V_s, W_s):
            return (U_s, V_s, W_s)

        return None

    # Structured Winograd-style initialization for better convergence
    def winograd_init(R, seed):
        np.random.seed(seed)
        # Initialize with structured values that might help
        U = np.zeros((N, R))
        V = np.zeros((M, R))
        W = np.zeros((P, R))
        for r in range(R):
            # Random structured pattern
            U[:, r] = np.random.choice([0, 0.5, 1.0, -0.5, -1.0], size=N)
            V[:, r] = np.random.choice([0, 0.5, 1.0, -0.5, -1.0], size=M)
            W[:, r] = np.random.choice([0, 0.5, 1.0, -0.5, -1.0], size=P)
        return U, V, W

    # Try ranks from 25 upward
    best_rank = N * M * P  # 36 fallback
    best_U = None
    best_V = None
    best_W = None

    for R in range(25, 37):
        if timed_out():
            break

        # Try multiple seeds with different initializations
        for seed in range(20):
            if timed_out():
                break

            result = try_als(R, 1000 + R * 100 + seed)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = R
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break

        if best_U is not None:
            break

    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook rank 36
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

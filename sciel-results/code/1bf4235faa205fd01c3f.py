import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n, m, p = 2, 3, 6
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U,V,W correctly decompose the 2x3x6 tensor."""
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((num_A, rank))
        V = np.zeros((num_B, rank))
        W = np.zeros((num_C, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W, rank

    def build_rank_33():
        """Build rank 33 decomposition using Strassen for 2x2x2 blocks."""
        rank = 33
        U = np.zeros((num_A, rank))
        V = np.zeros((num_B, rank))
        W = np.zeros((num_C, rank))
        col = 0

        # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> C[:, 0:6]
        for k in range(p):
            for i in range(n):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

        # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) -> C[:, 0:6]
        for t in range(3):
            # Strassen M1: (A00+A11)(B00+B11)
            U[0*m + 1, col] += 1.0
            U[1*m + 2, col] += 1.0
            V[1*p + 2*t, col] += 1.0
            V[2*p + 2*t + 1, col] += 1.0
            W[2*t*n + 0, col] += 1.0
            W[(2*t + 1)*n + 1, col] += 1.0
            col += 1

            # Strassen M2: (A10+A11)B00
            U[1*m + 1, col] += 1.0
            U[1*m + 2, col] += 1.0
            V[1*p + 2*t, col] += 1.0
            W[2*t*n + 1, col] += 1.0
            W[(2*t + 1)*n + 1, col] += -1.0
            col += 1

            # Strassen M3: A00(B01-B11)
            U[0*m + 1, col] += 1.0
            V[1*p + 2*t + 1, col] += 1.0
            V[2*p + 2*t + 1, col] += -1.0
            W[(2*t + 1)*n + 0, col] += 1.0
            W[(2*t + 1)*n + 1, col] += 1.0
            col += 1

            # Strassen M4: A11(B10-B00)
            U[1*m + 2, col] += 1.0
            V[2*p + 2*t, col] += 1.0
            V[1*p + 2*t, col] += -1.0
            W[2*t*n + 0, col] += 1.0
            W[2*t*n + 1, col] += 1.0
            col += 1

            # Strassen M5: (A00+A01)B11
            U[0*m + 1, col] += 1.0
            U[0*m + 2, col] += 1.0
            V[2*p + 2*t + 1, col] += 1.0
            W[2*t*n + 0, col] += -1.0
            W[(2*t + 1)*n + 0, col] += 1.0
            col += 1

            # Strassen M6: (A10-A00)(B00+B01)
            U[1*m + 1, col] += 1.0
            U[0*m + 1, col] += -1.0
            V[1*p + 2*t, col] += 1.0
            V[1*p + 2*t + 1, col] += 1.0
            W[(2*t + 1)*n + 1, col] += 1.0
            col += 1

            # Strassen M7: (A01-A11)(B10+B11)
            U[0*m + 2, col] += 1.0
            U[1*m + 2, col] += -1.0
            V[2*p + 2*t, col] += 1.0
            V[2*p + 2*t + 1, col] += 1.0
            W[2*t*n + 0, col] += 1.0
            col += 1

        return U, V, W, rank

    def try_optimize_rank(target_rank, init_U=None, init_V=None, init_W=None, n_restarts=20, max_nfev=3000):
        """Try to find a decomposition of given rank using numerical optimization."""
        if time.time() > deadline - 15:
            return None

        from scipy.optimize import least_squares

        # Build target tensor
        T_flat = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_flat[idx] = 1.0
                    idx += 1

        def objective(params):
            U = params[0:num_A*target_rank].reshape(num_A, target_rank)
            V = params[num_A*target_rank:num_A*target_rank + num_B*target_rank].reshape(num_B, target_rank)
            W = params[num_A*target_rank + num_B*target_rank:].reshape(num_C, target_rank)

            result = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        result[idx] = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                        idx += 1
            return result - T_flat

        best_params = None
        best_err = float('inf')

        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break

            if init_U is not None and restart == 0:
                if target_rank < 33:
                    U_init = init_U[:, :target_rank]
                    V_init = init_V[:, :target_rank]
                    W_init = init_W[:, :target_rank]
                    params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                elif target_rank > 33:
                    U_init = np.zeros((num_A, target_rank))
                    V_init = np.zeros((num_B, target_rank))
                    W_init = np.zeros((num_C, target_rank))
                    U_init[:, :33] = init_U
                    V_init[:, :33] = init_V
                    W_init[:, :33] = init_W
                    params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                else:
                    params0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            else:
                np.random.seed(42 + restart * 17)
                scale = 0.1 + 0.05 * (restart % 10)
                params0 = np.random.randn((num_A + num_B + num_C) * target_rank) * scale

            try:
                res = least_squares(objective, params0, method='lm', max_nfev=max_nfev)
                err = np.linalg.norm(res.fun)
                if err < best_err:
                    best_err = err
                    best_params = res.x.copy()
            except Exception:
                pass

        if best_params is None or best_err > 1e-8:
            return None

        U = best_params[0:num_A*target_rank].reshape(num_A, target_rank)
        V = best_params[num_A*target_rank:num_A*target_rank + num_B*target_rank].reshape(num_B, target_rank)
        W = best_params[num_A*target_rank + num_B*target_rank:].reshape(num_C, target_rank)

        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0

        if verify_decomposition(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap, target_rank

        if verify_decomposition(U, V, W):
            return U, V, W, target_rank

        return None

    # Build initial solutions
    U_school, V_school, W_school, rank_school = build_schoolbook()
    U_33, V_33, W_33, rank_33 = build_rank_33()

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    if verify_decomposition(U_33, V_33, W_33):
        best_rank = rank_33
        best_U = U_33.copy()
        best_V = V_33.copy()
        best_W = W_33.copy()

    # Try to find lower rank decompositions
    for target_rank in [32, 31, 30]:
        if time.time() > deadline - 20:
            break

        # Strategy 1: Direct optimization from truncated rank-33
        result = try_optimize_rank(target_rank, init_U=best_U, init_V=best_V, init_W=best_W,
                                   n_restarts=15, max_nfev=2000)
        if result is not None:
            U_new, V_new, W_new, rank_new = result
            if rank_new < best_rank:
                best_rank = rank_new
                best_U = U_new.copy()
                best_V = V_new.copy()
                best_W = W_new.copy()
                break

        # Strategy 2: Deflation - zero out random columns from rank-33 and optimize
        if time.time() > deadline - 15:
            break
        for attempt in range(10):
            if time.time() > deadline - 15:
                break
            np.random.seed(1000 + target_rank * 100 + attempt)
            cols_to_keep = np.random.choice(rank_33, target_rank, replace=False)
            U_def = U_33[:, cols_to_keep].copy()
            V_def = V_33[:, cols_to_keep].copy()
            W_def = W_33[:, cols_to_keep].copy()
            result = try_optimize_rank(target_rank, init_U=U_def, init_V=V_def, init_W=W_def,
                                       n_restarts=5, max_nfev=1500)
            if result is not None:
                U_new, V_new, W_new, rank_new = result
                if rank_new < best_rank:
                    best_rank = rank_new
                    best_U = U_new.copy()
                    best_V = V_new.copy()
                    best_W = W_new.copy()
                    break
            if result is not None:
                break

        if best_rank < target_rank + 1:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

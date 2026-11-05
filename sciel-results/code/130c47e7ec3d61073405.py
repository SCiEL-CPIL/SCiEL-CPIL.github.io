import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 6
    num_A = n * m      # 6
    num_B = m * p      # 18
    num_C = p * n      # 12

    def verify_decomposition(U, V, W, tol=1e-9):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_r[i, j, :], np.dot(V_r[j, k, :], W_r[k, i, :]))
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_rank_33():
        """Rank 33: schoolbook for col 0 (12) + Strassen for 3 blocks of 2x2x2 (21)."""
        rank = 33
        U = np.zeros((num_A, rank))
        V = np.zeros((num_B, rank))
        W = np.zeros((num_C, rank))
        col = 0

        # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> rank 12
        for k in range(p):
            for i in range(n):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

        # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) -> 3 Strassen blocks, rank 21
        for t in range(3):
            c0 = 2 * t
            c1 = 2 * t + 1

            # Strassen M1: (A00+A11)(B00+B11)
            U[0*m + 1, col] += 1.0
            U[1*m + 2, col] += 1.0
            V[1*p + c0, col] += 1.0
            V[2*p + c1, col] += 1.0
            W[c0*n + 0, col] += 1.0
            W[c1*n + 1, col] += 1.0
            col += 1

            # Strassen M2: (A10+A11)B00
            U[1*m + 1, col] += 1.0
            U[1*m + 2, col] += 1.0
            V[1*p + c0, col] += 1.0
            W[c0*n + 1, col] += 1.0
            W[c1*n + 1, col] += -1.0
            col += 1

            # Strassen M3: A00(B01-B11)
            U[0*m + 1, col] += 1.0
            V[1*p + c1, col] += 1.0
            V[2*p + c1, col] += -1.0
            W[c1*n + 0, col] += 1.0
            W[c1*n + 1, col] += 1.0
            col += 1

            # Strassen M4: A11(B10-B00)
            U[1*m + 2, col] += 1.0
            V[2*p + c0, col] += 1.0
            V[1*p + c0, col] += -1.0
            W[c0*n + 0, col] += 1.0
            W[c0*n + 1, col] += 1.0
            col += 1

            # Strassen M5: (A00+A01)B11
            U[0*m + 1, col] += 1.0
            U[0*m + 2, col] += 1.0
            V[2*p + c1, col] += 1.0
            W[c0*n + 0, col] += -1.0
            W[c1*n + 0, col] += 1.0
            col += 1

            # Strassen M6: (A10-A00)(B00+B01)
            U[1*m + 1, col] += 1.0
            U[0*m + 1, col] += -1.0
            V[1*p + c0, col] += 1.0
            V[1*p + c1, col] += 1.0
            W[c1*n + 1, col] += 1.0
            col += 1

            # Strassen M7: (A01-A11)(B10+B11)
            U[0*m + 2, col] += 1.0
            U[1*m + 2, col] += -1.0
            V[2*p + c0, col] += 1.0
            V[2*p + c1, col] += 1.0
            W[c0*n + 0, col] += 1.0
            col += 1

        return U, V, W

    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((num_A, rank))
        V = np.zeros((num_B, rank))
        W = np.zeros((num_C, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def objective(params, target_rank):
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
        return result - np.ones(n*m*p)

    def try_optimize_rank(target_rank, init_U, init_V, init_W, n_restarts=30, max_nfev=5000):
        if time.time() > deadline - 20:
            return None

        T_flat = np.ones(n * m * p)
        best_params = None
        best_err = float('inf')

        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break

            if restart == 0:
                U_init = np.zeros((num_A, target_rank))
                V_init = np.zeros((num_B, target_rank))
                W_init = np.zeros((num_C, target_rank))
                r = min(target_rank, init_U.shape[1])
                U_init[:, :r] = init_U[:, :r]
                V_init[:, :r] = init_V[:, :r]
                W_init[:, :r] = init_W[:, :r]
                params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                np.random.seed(1000 + restart * 31)
                scale = 0.05 + 0.03 * (restart % 10)
                params0 = np.random.randn((num_A + num_B + num_C) * target_rank) * scale

            try:
                res = least_squares(objective, params0, args=(target_rank,), method='lm', max_nfev=max_nfev)
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

        # Try snapping to half-integers
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        if verify_decomposition(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap

        if verify_decomposition(U, V, W, tol=1e-6):
            return U, V, W

        return None

    # Build rank-33 baseline
    U_33, V_33, W_33 = build_rank_33()

    if not verify_decomposition(U_33, V_33, W_33):
        U_s, V_s, W_s = build_schoolbook()
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    best_U, best_V, best_W = U_33, V_33, W_33

    # Try rank 32
    if time.time() < deadline - 25:
        result = try_optimize_rank(32, U_33, V_33, W_33, n_restarts=40, max_nfev=4000)
        if result is not None:
            best_U, best_V, best_W = result

    # Try rank 31
    if best_U.shape[1] > 31 and time.time() < deadline - 20:
        result = try_optimize_rank(31, best_U, best_V, best_W, n_restarts=30, max_nfev=3000)
        if result is not None:
            best_U, best_V, best_W = result

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

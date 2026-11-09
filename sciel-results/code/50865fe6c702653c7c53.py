import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n

    start_time = time.time()
    time_limit = 140.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n_rows_U, rank))
        V = np.zeros((n_rows_V, rank))
        W = np.zeros((n_rows_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    best_U, best_V, best_W, best_rank = schoolbook()

    def check_error(U, V, W, r):
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        T1 = np.einsum('ijc,kic->ikjc', U_t, W_t)
        E = np.einsum('ikjc,lkc->ikjl', T1, V_t)
        D = np.eye(m)
        Target = np.zeros((n, p, m, m))
        Target[:, :, :, :] = D
        return float(np.sum((E - Target) ** 2))

    def snap_half(x):
        return np.round(x * 2) / 2

    def try_rank(r, max_restarts, iters):
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                return None
            U = np.random.randn(n_rows_U, r) * 0.3
            V = np.random.randn(n_rows_V, r) * 0.3
            W = np.random.randn(n_rows_W, r) * 0.3
            for it in range(iters):
                if time.time() - start_time > time_limit:
                    return None
                U_t = U.reshape(n, m, r)
                V_t = V.reshape(m, p, r)
                W_t = W.reshape(p, n, r)
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.einsum('kic,lkc->klc', W_t, V_t).reshape(p * m, r)
                        t = np.zeros(p * m)
                        for k in range(p):
                            t[k * m + j] = 1.0
                        sol, *_ = np.linalg.lstsq(M, t, rcond=None)
                        U[i * m + j, :] = sol
                        U_t[i, j, :] = sol
                # Update V
                for l in range(m):
                    for k in range(p):
                        M = np.einsum('ijc,kic->ikjc', U_t, W_t).reshape(n * m, r)
                        t = np.zeros(n * m)
                        for i in range(n):
                            t[i * m + l] = 1.0
                        sol, *_ = np.linalg.lstsq(M, t, rcond=None)
                        V[l * p + k, :] = sol
                        V_t[l, k, :] = sol
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.einsum('ijc,lkc->jlc', U_t, V_t).reshape(m * m, r)
                        t = np.zeros(m * m)
                        for j in range(m):
                            t[j * m + j] = 1.0
                        sol, *_ = np.linalg.lstsq(M, t, rcond=None)
                        W[k * n + i, :] = sol
                        W_t[k, i, :] = sol
            err = check_error(U, V, W, r)
            if err < 1e-8:
                U_s, V_s, W_s = snap_half(U), snap_half(V), snap_half(W)
                if check_error(U_s, V_s, W_s, r) < 1e-8:
                    return U_s, V_s, W_s, r
                if (np.allclose(U, U_s, atol=1e-4) and
                    np.allclose(V, V_s, atol=1e-4) and
                    np.allclose(W, W_s, atol=1e-4)):
                    return U_s, V_s, W_s, r
        return None

    # Search for rank below schoolbook, minimal first
    candidate_ranks = [25, 30, 40, 50, 60, 65, 70]
    for r in candidate_ranks:
        if r >= best_rank:
            continue
        if time.time() - start_time > time_limit - 15:
            break
        res = try_rank(r, max_restarts=2, iters=25)
        if res is not None:
            U, V, W, rank = res
            if rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, rank
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

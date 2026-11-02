"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _solve_for_W(U, V, n, m, p, R):
    W = np.zeros((p * n, R), dtype=np.float64)
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    for k in range(p):
        for i in range(n):
            A = U_3d[i, :, :] * V_3d[:, k, :]
            try:
                x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W[k * n + i, :] = x
            except:
                pass
    return W

def _solve_for_U(V, W, n, m, p, R):
    U = np.zeros((n * m, R), dtype=np.float64)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    for i in range(n):
        for j in range(m):
            A = V_3d[j, :, :] * W_3d[:, i, :]
            try:
                x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U[i * m + j, :] = x
            except:
                pass
    return U

def _solve_for_V(U, W, n, m, p, R):
    V = np.zeros((m * p, R), dtype=np.float64)
    U_3d = U.reshape(n, m, R)
    W_3d = W.reshape(p, n, R)
    for j in range(m):
        for k in range(p):
            A = U_3d[:, j, :] * W_3d[k, :, :]
            try:
                x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V[j * p + k, :] = x
            except:
                pass
    return V

def _verify(U, V, W, n, m, p, R, tol=1e-8):
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _try_rank(R, max_iters=100, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    n, m, p = DIMENSIONS
    U = np.random.randn(n * m, R) * 0.5
    V = np.random.randn(m * p, R) * 0.5
    W = np.random.randn(p * n, R) * 0.5
    for it in range(max_iters):
        W = _solve_for_W(U, V, n, m, p, R)
        U = _solve_for_U(V, W, n, m, p, R)
        V = _solve_for_V(U, W, n, m, p, R)
        if _verify(U, V, W, n, m, p, R, tol=1e-8):
            return U, V, W, True
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    for R in [15, 20, 25, 30, 40, 50, 60]:
        if time.time() > deadline:
            break
        for seed in range(10):
            if time.time() > deadline:
                break
            U, V, W, success = _try_rank(R, max_iters=100, seed=seed)
            if success:
                U_list = [[round(float(x), 6) for x in row] for row in U]
                V_list = [[round(float(x), 6) for x in row] for row in V]
                W_list = [[round(float(x), 6) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
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
# EVOLVE-BLOCK-END

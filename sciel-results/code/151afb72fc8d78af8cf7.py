"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import time

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 125

    # Build the matrix multiplication tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback
    rank_school = n * m * p
    U_fallback = np.zeros((n * m, rank_school))
    V_fallback = np.zeros((m * p, rank_school))
    W_fallback = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j, col] = 1.0
                V_fallback[j * p + k, col] = 1.0
                W_fallback[k * n + i, col] = 1.0
                col += 1

    def verify(U, V, W, tol=1e-9):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def snap_halfint(X, tol=1e-6):
        X2 = 2.0 * X
        Xr = np.round(X2)
        if np.max(np.abs(X2 - Xr)) < tol:
            return Xr / 2.0
        return None

    def try_als(rank, n_iter=30, seed=0):
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank) * 0.01
        V = rng.randn(m * p, rank) * 0.01
        W = rng.randn(p * n, rank) * 0.01

        for it in range(n_iter):
            if time.time() > deadline:
                return None

            A_U = np.zeros((m * p * p * n, rank))
            for r in range(rank):
                A_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            for a in range(n * m):
                rhs = T[a, :, :].flatten()
                U[a, :] = np.linalg.lstsq(A_U, rhs, rcond=None)[0]

            A_V = np.zeros((n * m * p * n, rank))
            for r in range(rank):
                A_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            for b in range(m * p):
                rhs = T[:, b, :].flatten()
                V[b, :] = np.linalg.lstsq(A_V, rhs, rcond=None)[0]

            A_W = np.zeros((n * m * m * p, rank))
            for r in range(rank):
                A_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            for c in range(p * n):
                rhs = T[:, :, c].flatten()
                W[c, :] = np.linalg.lstsq(A_W, rhs, rcond=None)[0]

            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < 1e-12:
                break

        U_s = snap_halfint(U, 1e-3)
        V_s = snap_halfint(V, 1e-3)
        W_s = snap_halfint(W, 1e-3)
        if U_s is not None and V_s is not None and W_s is not None:
            if verify(U_s, V_s, W_s):
                return U_s, V_s, W_s
        return None

    best_result = None

    for rank in range(55, 63):
        if time.time() > deadline:
            break
        for seed in range(5):
            if time.time() > deadline:
                break
            result = try_als(rank, n_iter=20, seed=seed)
            if result is not None:
                best_result = result
                break
        if best_result is not None:
            break

    if best_result is None:
        U_out = U_fallback.tolist()
        V_out = V_fallback.tolist()
        W_out = W_fallback.tolist()
    else:
        U_out = best_result[0].tolist()
        V_out = best_result[1].tolist()
        W_out = best_result[2].tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END


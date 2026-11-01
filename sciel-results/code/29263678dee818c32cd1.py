"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_error(U, V, W, n, m, p, rank):
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _check_valid(U, V, W, n, m, p, rank, tol=1e-8):
    return _compute_error(U, V, W, n, m, p, rank) < tol

def _solve_for_W(U, V, n, m, p, rank):
    W = np.zeros((p*n, rank))
    for k in range(p):
        for i in range(n):
            A = np.array([[U[i*m+j, r] * V[j*p+k, r] for r in range(rank)] for j in range(m)])
            b = np.ones(m)
            w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[k*n+i, :] = w
    return W

def _solve_for_V(U, W, n, m, p, rank):
    V = np.zeros((m*p, rank))
    for j in range(m):
        for k in range(p):
            A = np.array([[U[i*m+j, r] * W[k*n+i, r] for r in range(rank)] for i in range(n)])
            b = np.ones(n)
            w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V[j*p+k, :] = w
    return V

def _solve_for_U(V, W, n, m, p, rank):
    U = np.zeros((n*m, rank))
    for i in range(n):
        for j in range(m):
            A = np.array([[V[j*p+k, r] * W[k*n+i, r] for r in range(rank)] for k in range(p)])
            b = np.ones(p)
            w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U[i*m+j, :] = w
    return U

def _alternating(rank, n, m, p, max_iter=500, seed=42, time_limit=5.0):
    start = time.time()
    np.random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.3
    V = np.random.randn(m*p, rank) * 0.3
    W = np.random.randn(p*n, rank) * 0.3
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        U = _solve_for_U(V, W, n, m, p, rank)
        V = _solve_for_V(U, W, n, m, p, rank)
        W = _solve_for_W(U, V, n, m, p, rank)
        err = _compute_error(U, V, W, n, m, p, rank)
        if err < 1e-9:
            break
    
    if err < 1e-8:
        return U, V, W
    return None

def _try_round(U, V, W, n, m, p, rank, tol=1e-6):
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    if _check_valid(U_r, V_r, W_r, n, m, p, rank, tol=tol):
        return U_r, V_r, W_r
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    for r in [10, 11, 12, 13, 14, 15, 16, 18, 20]:
        if time.time() > deadline - 15:
            break
        for seed in range(30):
            if time.time() > deadline - 10:
                break
            result = _alternating(r, n, m, p, max_iter=500, seed=seed, time_limit=3.0)
            if result is not None:
                U, V, W = result
                rounded = _try_round(U, V, W, n, m, p, r)
                if rounded is not None:
                    U, V, W = rounded
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
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

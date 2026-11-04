"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _compute_error(n, m, p, U, V, W):
    """Compute max |T[i,j,k] - 1| over all entries."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _als_update_U(n, m, p, U, V, W):
    rank = U.shape[1]
    new_U = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            A = np.empty((p, rank))
            for k in range(p):
                A[k, :] = V[j*p+k] * W[k*n+i]
            try:
                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                new_U[i*m+j] = u
            except np.linalg.LinAlgError:
                new_U[i*m+j] = U[i*m+j]
    return new_U

def _als_update_V(n, m, p, U, V, W):
    rank = V.shape[1]
    new_V = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            A = np.empty((n, rank))
            for i in range(n):
                A[i, :] = U[i*m+j] * W[k*n+i]
            try:
                v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                new_V[j*p+k] = v
            except np.linalg.LinAlgError:
                new_V[j*p+k] = V[j*p+k]
    return new_V

def _als_update_W(n, m, p, U, V, W):
    rank = W.shape[1]
    new_W = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            A = np.empty((m, rank))
            for j in range(m):
                A[j, :] = U[i*m+j] * V[j*p+k]
            try:
                w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                new_W[k*n+i] = w
            except np.linalg.LinAlgError:
                new_W[k*n+i] = W[k*n+i]
    return new_W

def _polish(n, m, p, U, V, W, deadline):
    """Final polish with L-BFGS-B to drive error to zero."""
    rank = U.shape[1]
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def objective(params):
        Uf = params[:n*m*rank].reshape(n*m, rank)
        Vf = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        Wf = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Uf[i*m+j], Vf[j*p+k] * Wf[k*n+i])
                    err += (s - 1.0) ** 2
        return err

    try:
        res = minimize(objective, params0, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14})
        if res.fun < 1e-14:
            Uf = res.x[:n*m*rank].reshape(n*m, rank)
            Vf = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            Wf = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            return Uf, Vf, Wf
    except Exception:
        pass
    return None

def _try_rank(n, m, p, rank, max_iters=200, n_restarts=4, seed=42, deadline=None):
    best_error = float('inf')
    best_result = None

    for restart in range(n_restarts):
        if deadline and time.time() > deadline - 12:
            break
        np.random.seed(seed + restart * 137)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)

        for it in range(max_iters):
            if deadline and time.time() > deadline - 5:
                break
            U = _als_update_U(n, m, p, U, V, W)
            V = _als_update_V(n, m, p, U, V, W)
            W = _als_update_W(n, m, p, U, V, W)
            err = _compute_error(n, m, p, U, V, W)
            if err < best_error:
                best_error = err
                best_result = (U.copy(), V.copy(), W.copy())
            if err < 1e-10:
                break

    if best_result is None:
        return None
    U, V, W = best_result
    if best_error < 1e-6:
        polished = _polish(n, m, p, U, V, W, deadline)
        if polished is not None:
            U, V, W = polished
            if _compute_error(n, m, p, U, V, W) < 1e-8:
                return U.tolist(), V.tolist(), W.tolist()
        elif best_error < 1e-8:
            return U.tolist(), V.tolist(), W.tolist()
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110

    # Try ranks from 28 down to 20
    for rank in range(28, 19, -1):
        if time.time() > deadline - 25:
            break
        result = _try_rank(n, m, p, rank, max_iters=150, n_restarts=4,
                           seed=rank * 31 + 7, deadline=deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END

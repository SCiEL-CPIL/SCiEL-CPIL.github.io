"""Program attempting to find a low-rank tensor decomposition for 3x5 * 5x6 matrix multiplication.

The tensor T has indices (i,j,k) with i in [0,3), j in [0,5), k in [0,6).
U has shape (15, r), V has shape (30, r), W has shape (18, r).
The decomposition satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).

This program uses a two-phase approach:
1. Random Gaussian initialization with greedy rank augmentation and Newton's method
2. Integer/half-integer rounding with exact verification

If no valid low-rank solution is found within the time limit, returns schoolbook (rank 90).
"""

import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]
N, M, P = 3, 5, 6
NM = N * M  # 15
MP = M * P  # 30
PN = P * N  # 18
TENSOR_SIZE = NM * MP * PN  # 15 * 30 * 18 = 8100

DEADLINE = 135.0  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _verify_exact(U, V, W):
    """Verify that the decomposition exactly equals the tensor (all entries = 1).
    Returns True if exact, False otherwise.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    # Compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for all (i,j,k)
    # Reshape to tensors
    Ut = U.reshape(N, M, r)  # [i, j, c]
    Vt = V.reshape(M, P, r)  # [j, k, c]
    Wt = W.reshape(P, N, r)  # [k, i, c]
    # For each (i,j,k): sum_c Ut[i,j,c] * Vt[j,k,c] * Wt[k,i,c]
    # This is a 4D contraction. Let's do it carefully.
    # Result[i,j,k] = sum_c Ut[i,j,c] * Vt[j,k,c] * Wt[k,i,c]
    result = np.zeros((N, M, P), dtype=np.float64)
    for c in range(r):
        # Ut[:, :, c] is (N, M), Vt[:, :, c] is (M, P), Wt[:, :, c] is (P, N)
        uc = Ut[:, :, c]  # (N, M)
        vc = Vt[:, :, c]  # (M, P)
        wc = Wt[:, :, c]  # (P, N)
        # For each i,j,k: uc[i,j] * vc[j,k] * wc[k,i]
        # outer product approach:
        for i in range(N):
            for j in range(M):
                if uc[i, j] != 0:
                    # vc[j, :] is (P,), wc[:, i] is (P,)
                    # contribution to result[i, j, :] is uc[i,j] * vc[j,:] * wc[:,i]
                    result[i, j, :] += uc[i, j] * vc[j, :] * wc[:, i]
    # Check all entries are exactly 1 (within floating point tolerance)
    if not np.allclose(result, 1.0, atol=1e-10, rtol=1e-10):
        return False
    return True


def _try_round_to_half_int(U, V, W, tol=1e-8):
    """Try to round entries to nearest 0.5 and verify exactness.
    Returns (U_r, V_r, W_r) if successful, else None.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    # Check if rounding changed anything significantly
    if np.max(np.abs(U_r - U)) > 1e-6 or np.max(np.abs(V_r - V)) > 1e-6 or np.max(np.abs(W_r - W)) > 1e-6:
        return None
    # Verify exactness with rounded values
    if _verify_exact(U_r.tolist(), V_r.tolist(), W_r.tolist()):
        return (U_r.tolist(), V_r.tolist(), W_r.tolist())
    return None


def _newton_step(U, V, W, residual, lr=0.01, max_iter=50):
    """Perform a few Newton/LBFGS-style gradient steps to reduce residual.
    Uses simple gradient descent with line search on the residual norm.
    """
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    best_norm = np.linalg.norm(residual, 2)
    
    for it in range(max_iter):
        # Compute gradients
        # dL/dU[i,j,c] = sum_k V[j,k,c] * W[k,i,c] * (residual[i,j,k] - 1)
        # But our tensor target is 1 for all (i,j,k), so residual = current - 1
        # Actually let's define: f(i,j,k) = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # We want f = 1 everywhere.
        # Gradient of sum (f-1)^2 w.r.t. U[i,j,c]:
        # = 2 * sum_k (f[i,j,k]-1) * V[j,k,c] * W[k,i,c]
        
        # Compute f
        f = np.zeros((N, M, P))
        for c in range(r):
            uc = U[i*m+j, c]  # scalar
            vc = V[j*p+k, c]  # scalar
            wc = W[k*n+i, c]  # scalar
            # This loop is too slow. Use tensor approach.
            pass
        
        # Simpler: use the residual directly
        # grad_U[i*m+j, c] = sum_k residual[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(r):
            uc = U[:, c].reshape(N, M)
            vc = V[:, c].reshape(M, P)
            wc = W[:, c].reshape(P, N)
            for i in range(N):
                for j in range(M):
                    # grad_U[i,j,c] = sum_k res[i,j,k] * vc[j,k] * wc[k,i]
                    g = np.dot(residual[i, j, :], vc[j, :] * wc[:, i])
                    grad_U[i, j, c] = g
                    # grad_V[j,k,c] = sum_i res[i,j,k] * uc[i,j] * wc[k,i]
                for k in range(P):
                    gv = np.dot(residual[:, j, k], uc[:, j] * wc[k, :])
                    grad_V[j, k, c] = gv
                    # grad_W[k,i,c] = sum_j res[i,j,k] * uc[i,j] * vc[j,k]
            for k in range(P):
                for i in range(N):
                    gw = np.dot(residual[i, :, k], uc[i, :] * vc[:, k])
                    grad_W[k, i, c] = gw
        
        # Gradient descent step
        step = lr / max(1, it)
        U -= step * grad_U
        V -= step * grad_V
        W -= step * grad_W
        
        # Recompute residual
        new_res = np.zeros((N, M, P))
        for c in range(r):
            uc = U[:, c].reshape(N, M)
            vc = V[:, c].reshape(M, P)
            wc = W[:, c].reshape(P, N)
            for i in range(N):
                for j in range(M):
                    new_res[i, j, :] += uc[i, j] * vc[j, :] * wc[:, i]
        new_res -= 1.0
        new_norm = np.linalg.norm(new_res, 2)
        if new_norm < best_norm * 0.99:
            best_norm = new_norm
            residual = new_res
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    return U, V, W, residual, best_norm


def _greedy_rank_search(max_rank=70, n_restarts=5, time_budget=120.0):
    """Try to find a rank-r decomposition for r < 90.
    
    Strategy:
    - Start with rank r0 (e.g., 50) random Gaussian matrices
    - Run Newton iterations to minimize residual
    - If residual is small enough, try rounding to half-integers
    - If rounding fails, increment rank and try again
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_result = None
    best_rank = 91  # worse than schoolbook
    
    for rank in range(40, min(max_rank + 1, 89)):
        if _elapsed() > time_budget:
            break
        for restart in range(n_restarts):
            if _elapsed() > time_budget:
                break
            rng = np.random.RandomState(rank * 1000 + restart)
            U = rng.randn(NM, rank) * 0.1
            V = rng.randn(MP, rank) * 0.1
            W = rng.randn(PN, rank) * 0.1
            
            # Compute initial residual
            f = np.zeros((N, M, P))
            for c in range(rank):
                uc = U[:, c].reshape(N, M)
                vc = V[:, c].reshape(M, P)
                wc = W[:, c].reshape(P, N)
                for i in range(N):
                    for j in range(M):
                        f[i, j, :] += uc[i, j] * vc[j, :] * wc[:, i]
            residual = f - 1.0
            init_norm = np.linalg.norm(residual, 2)
            
            # Run Newton iterations
            U, V, W, residual, final_norm = _newton_step(
                U, V, W, residual, lr=0.005, max_iter=30
            )
            
            if final_norm < 1e-6:
                # Try rounding
                result = _try_round_to_half_int(U, V, W)
                if result is not None:
                    return result
                # If rounding didn't work but residual is tiny, maybe try more iterations
                U, V, W, residual, final_norm2 = _newton_step(
                    U, V, W, residual, lr=0.001, max_iter=20
                )
                if final_norm2 < 1e-8:
                    result = _try_round_to_half_int(U, V, W)
                    if result is not None:
                        return result
            
            if final_norm < best_rank * 0.01:  # heuristic threshold
                best_rank = rank
                best_result = (U.tolist(), V.tolist(), W.tolist())
    
    return None


def construct():
    """Construct a low-rank tensor decomposition for 3x5 * 5x6 matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' containing factor matrices.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Phase 1: Try to find a better-than-schoolbook decomposition
    result = _greedy_rank_search(max_rank=75, n_restarts=3, time_budget=120.0)
    
    if result is not None:
        return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Phase 2: Fall back to schoolbook decomposition (rank 90)
    n, m, p = N, M, P
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


_START_TIME = None

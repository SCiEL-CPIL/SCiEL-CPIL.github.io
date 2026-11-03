"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import numpy as np
import time
import random
from itertools import product

# EVOLVE-BLOCK-START
def _is_half_integer(x, tol=1e-9):
    """Check if x is close to a half-integer."""
    nearest = round(x * 2) / 2
    return abs(x - nearest) < tol

def _snap_to_half_integer(arr, tol=1e-9):
    """Snap array entries to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped, True
    return arr, False

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    # Reconstruct tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(len(U[0])):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_iterate(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """Alternating least squares for tensor decomposition."""
    rank = U.shape[1]
    for it in range(max_iter):
        # Update U
        U_old = U.copy()
        for idx in range(n * m):
            i, j = divmod(idx, m)
            # For each c, we need U[idx, c] such that sum_c U[idx,c] V[j*p+k,c] W[k*n+i,c] = 1 for all k
            # This is a linear system in U[idx, :]
            A = np.zeros((p, rank))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            b = np.ones(p)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                U[idx, :] = sol
            except:
                pass
        # Update V
        V_old = V.copy()
        for idx in range(m * p):
            j, k = divmod(idx, p)
            A = np.zeros((n, rank))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            b = np.ones(n)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                V[idx, :] = sol
            except:
                pass
        # Update W
        W_old = W.copy()
        for idx in range(p * n):
            k, i = divmod(idx, n)
            A = np.zeros((m, rank))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b = np.ones(m)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                W[idx, :] = sol
            except:
                pass
        # Check convergence
        if np.max(np.abs(U - U_old)) < tol and np.max(np.abs(V - V_old)) < tol and np.max(np.abs(W - W_old)) < tol:
            break
    return U, V, W

def _try_rank(n, m, p, rank, max_restarts=10, seed=None):
    """Try to find a rank-r exact decomposition."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    for restart in range(max_restarts):
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # ALS iterations
        U, V, W = _als_iterate(U, V, W, n, m, p, max_iter=100)
        
        # Check if close to exact
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-8:
            # Try to snap to half-integers
            U_snapped, u_ok = _snap_to_half_integer(U, tol=1e-6)
            V_snapped, v_ok = _snap_to_half_integer(V, tol=1e-6)
            W_snapped, w_ok = _snap_to_half_integer(W, tol=1e-6)
            if u_ok and v_ok and w_ok:
                if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                    return U_snapped, V_snapped, W_snapped
            else:
                # Try smaller tolerance for snapping
                U_snapped, u_ok = _snap_to_half_integer(U, tol=1e-4)
                V_snapped, v_ok = _snap_to_half_integer(V, tol=1e-4)
                W_snapped, w_ok = _snap_to_half_integer(W, tol=1e-4)
                if u_ok and v_ok and w_ok:
                    if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                        return U_snapped, V_snapped, W_snapped
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank
    # Known lower bound for all-ones tensor (n,m,p) is at least max(n,m,p)
    # Try ranks from a reasonable lower bound upward
    lower_bound = max(n, m, p)
    
    best_result = None
    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, max_restarts=5, seed=rank)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

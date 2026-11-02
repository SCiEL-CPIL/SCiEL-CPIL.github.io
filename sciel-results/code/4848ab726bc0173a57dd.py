import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i1 in range(n):
        for r in range(m):
            for j2 in range(p):
                i = i1 * m + r
                j = r * p + j2
                k = i1 * n + j2
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-6):
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.max(np.abs(recon - T))
    return err < tol

def _is_half_integer(mat, tol=1e-6):
    return np.allclose(2 * mat, np.round(2 * mat), atol=tol)

def _build_rank18():
    """Construct the known rank-18 decomposition."""
    n, m, p = DIMENSIONS
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1
        
        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1
        
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1
        
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1
        
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1
    
    return U, V, W

def _try_rank_search(target_rank, max_iters=50, attempts=10, time_limit=20):
    """Search for a decomposition of the given rank."""
    n, m, p = DIMENSIONS
    T = _build_target()
    R = target_rank
    start_time = time.time()
    deadline = start_time + time_limit
    
    for attempt in range(attempts):
        if time.time() > deadline:
            break
        
        np.random.seed(attempt * 1000 + 42)
        U = np.random.randint(-2, 3, size=(n*m, R)).astype(float)
        V = np.random.randint(-2, 3, size=(m*p, R)).astype(float)
        W = np.random.randint(-2, 3, size=(p*n, R)).astype(float)
        
        for r in range(R):
            if np.all(U[:, r] == 0): U[0, r] = 1
            if np.all(V[:, r] == 0): V[0, r] = 1
            if np.all(W[:, r] == 0): W[0, r] = 1
        
        for it in range(max_iters):
            # ALS step for U
            U_new = np.zeros_like(U)
            for i in range(n*m):
                b = T[i].flatten()
                A = np.zeros((m*p * p*n, R))
                for r in range(R):
                    outer = np.outer(V[:, r], W[:, r])
                    A[:, r] = outer.flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i] = x
                except:
                    U_new[i] = U[i]
            U = U_new
            
            # ALS step for V
            V_new = np.zeros_like(V)
            for j in range(m*p):
                b = T[:, j, :].transpose(1, 0).flatten()
                A = np.zeros((n*m * p*n, R))
                for r in range(R):
                    outer = np.outer(U[:, r], W[:, r])
                    A[:, r] = outer.flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j] = x
                except:
                    V_new[j] = V[j]
            V = V_new
            
            # ALS step for W
            W_new = np.zeros_like(W)
            for k in range(p*n):
                b = T[:, :, k].flatten()
                A = np.zeros((n*m * m*p, R))
                for r in range(R):
                    outer = np.outer(U[:, r], V[:, r])
                    A[:, r] = outer.flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k] = x
                except:
                    W_new[k] = W[k]
            W = W_new
            
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(recon - T))
            if err < 1e-5:
                U_rounded = np.round(2 * U) / 2
                V_rounded = np.round(2 * V) / 2
                W_rounded = np.round(2 * W) / 2
                if _check_decomposition(U_rounded, V_rounded, W_rounded, T):
                    return U_rounded, V_rounded, W_rounded
                if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                    if _check_decomposition(U, V, W, T):
                        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target()
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Try rank 13
    if time.time() < deadline - 60:
        result = _try_rank_search(13, max_iters=30, attempts=5, time_limit=30)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Try rank 14
    if time.time() < deadline - 40:
        result = _try_rank_search(14, max_iters=30, attempts=5, time_limit=20)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Try rank 15
    if time.time() < deadline - 20:
        result = _try_rank_search(15, max_iters=30, attempts=5, time_limit=15)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to Rank 18
    U, V, W = _build_rank18()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

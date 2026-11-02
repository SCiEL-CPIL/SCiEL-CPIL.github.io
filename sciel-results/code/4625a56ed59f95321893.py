"""Matrix multiplication 4x4x6 - rank-minimal tensor decomposition."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random

def _verify_rank(n, m, p, rank, U, V, W):
    """Verify that the given factors decompose the matmul tensor correctly.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    Convention: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c = 1 for all i,j,k
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > 1e-6:
                    return False, (i, j, k, val)
    return True, None

def _schoolbook(n, m, p):
    """Construct the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
    return U, V, W, rank

def _find_decomposition(n, m, p, target_rank, max_time=100.0, seed=42):
    """Try to find a tensor decomposition of the matmul tensor at target_rank.
    Uses iterative least-squares alternating minimization.
    Returns (U, V, W, success)
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    np.random.seed(seed)
    random.seed(seed)
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(30):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        U = np.random.randn(n_m, target_rank) * 0.3
        V = np.random.randn(m_p, target_rank) * 0.3
        W = np.random.randn(p_n, target_rank) * 0.3
        
        # Normalize each column
        for c in range(target_rank):
            nu = np.linalg.norm(U[:, c])
            nv = np.linalg.norm(V[:, c])
            nw = np.linalg.norm(W[:, c])
            if nu * nv * nw > 1e-12:
                scale = (1.0 / (nu * nv * nw)) ** (1.0/3.0)
                U[:, c] *= scale
                V[:, c] *= scale
                W[:, c] *= scale
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # Alternate: optimize W given U, V
            for i in range(n):
                for k in range(p):
                    # Need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                    # Let A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # Then: A @ W[k*n+i, :] = ones(m)
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n + i, :] = sol
            
            # Optimize U given V, W
            for i in range(n):
                for j in range(m):
                    # Need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                    # Let B[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Then: B @ U[i*m+j, :] = ones(p)
                    B = np.zeros((p, target_rank))
                    for k in range(p):
                        B[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    sol, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                    U[i*m + j, :] = sol
            
            # Optimize V given U, W
            for j in range(m):
                for k in range(p):
                    # Need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                    # Let C[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    # Then: C @ V[j*p+k, :] = ones(n)
                    C = np.zeros((n, target_rank))
                    for i in range(n):
                        C[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    sol, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
                    V[j*p + k, :] = sol
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-8:
                return best_U, best_V, best_W, True
        
        # If we found a good solution, return it
        if best_err < 1e-6:
            return best_U, best_V, best_W, True
    
    # Return best found
    if best_U is not None and best_err < 1e-4:
        return best_U, best_V, best_W, True
    return None, None, None, False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline, with margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb, rank_fb = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Known bounds: rank(4x4x6) is between some lower and upper bounds
    # Theoretical lower bound is around 4*4*6 / something, but let's try practical values
    # Try ranks from 80 down to 60
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fb
    
    # Try various target ranks
    for target_rank in [80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60]:
        if time.time() > deadline - 20:
            break
        
        U, V, W, success = _find_decomposition(n, m, p, target_rank, 
                                                 max_time=min(15, deadline - time.time() - 10),
                                                 seed=42 + target_rank)
        if success:
            # Verify
            ok, info = _verify_rank(n, m, p, target_rank, U, V, W)
            if ok:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

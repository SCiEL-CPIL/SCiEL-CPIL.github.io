"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for c in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:,c], V_arr[:,c], W_arr[:,c])
    
    T_target = _build_tensor(n, m, p)
    
    if not np.allclose(T_decomp, T_target, atol=1e-8):
        return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _als_decompose(T, n, m, p, rank, max_iter=100, seed=0):
    """
    Alternating Least Squares for tensor decomposition.
    T is the target tensor (n*m, m*p, p*n).
    """
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for _ in range(max_iter):
        # Solve for U
        A = np.zeros((m*p * p*n, rank))
        B = np.zeros((m*p * p*n, n*m))
        
        idx = 0
        for j in range(m*p):
            for k in range(p*n):
                A[idx, :] = V[:, j] * W[:, k]
                B[idx, :] = T[:, j, k]
                idx += 1
                
        U_sol, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
        U = U_sol.T
        
        # Solve for V
        A = np.zeros((n*m * p*n, rank))
        B = np.zeros((n*m * p*n, m*p))
        idx = 0
        for i in range(n*m):
            for k in range(p*n):
                A[idx, :] = U[i, :] * W[k, :]
                B[idx, :] = T[i, :, k]
                idx += 1
        V_sol, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
        V = V_sol.T
        
        # Solve for W
        A = np.zeros((n*m * m*p, rank))
        B = np.zeros((n*m * m*p, p*n))
        idx = 0
        for i in range(n*m):
            for j in range(m*p):
                A[idx, :] = U[i, :] * V[j, :]
                B[idx, :] = T[i, j, :]
                idx += 1
        W_sol, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
        W = W_sol.T
        
    return U, V, W

def _find_decomposition(n, m, p, deadline):
    T = _build_tensor(n, m, p)
    # Try ranks from 44 down to 30
    for r in range(44, 29, -1):
        if time.time() > deadline - 10:
            break
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            try:
                U, V, W = _als_decompose(T, n, m, p, r, max_iter=50, seed=seed)
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Convert to lists
                def to_list(mat):
                    return [[int(round(v)) if abs(v - round(v)) < 1e-9 else v for v in row] for row in mat]
                
                U_list = to_list(U_rounded)
                V_list = to_list(V_rounded)
                W_list = to_list(W_rounded)
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list, "rank": r}
            except Exception:
                continue
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135
    
    # Try to find a better decomposition
    result = _find_decomposition(n, m, p, deadline)
    if result:
        return result
        
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END

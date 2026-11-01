import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
n, m, p = DIMENSIONS
N = n * m  # 6
M = m * p  # 9
P = p * n  # 6
SCHOOLBOOK = n * m * p  # 18

def build_tensor():
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that U, V, W exactly decompose T."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_r = np.array(U).reshape(n, m, rank)
    V_r = np.array(V).reshape(m, p, rank)
    W_r = np.array(W).reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                if abs(val - T[i, j, k]) > tol:
                    return False
    return True

def try_rank_structured(r, T, max_iters=2000, seed=42, time_limit=8.0):
    """Try to find a rank-r decomposition using structured initialization and refinement."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    for attempt in range(10):
        # Structured initialization: try to exploit the all-ones structure
        if attempt == 0:
            # Try a block-structured initialization
            U = np.zeros((N, r))
            V = np.zeros((M, r))
            W = np.zeros((P, r))
            
            # Try to assign each (i,j,k) to a subset of rank components
            # For simplicity, try cyclic assignment
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Assign to component (i*m+j)*p+k mod r
                        c = (i * m + j) * p + k
                        c = c % r
                        U[i*m+j, c] += 1.0
                        V[j*p+k, c] += 1.0
                        W[k*n+i, c] += 1.0
        elif attempt == 1:
            # Sparse initialization
            U = np.zeros((N, r))
            V = np.zeros((M, r))
            W = np.zeros((P, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = (i * m + j) * p + k
                        c = c % r
                        U[i*m+j, c] = 1.0
                        V[j*p+k, c] = 1.0
                        W[k*n+i, c] = 1.0
        else:
            # Random initialization with scaling
            U = rng.randn(N, r) * (1.0 / np.sqrt(r))
            V = rng.randn(M, r) * (1.0 / np.sqrt(r))
            W = rng.randn(P, r) * (1.0 / np.sqrt(r))
        
        best_obj = np.inf
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Alternating least squares with vectorized operations
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((r, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    b = T[i, j, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((r, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    b = T[:, j, k]
                    try:
                        v, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((r, m))
                    for j in range(m):
                        A[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    b = T[i, :, k]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Calculate objective
            obj = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        obj += (val - T[i,j,k])**2
            
            if obj < 1e-14:
                return U, V, W
            
            if obj < best_obj:
                best_obj = obj
    
    return None

def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s safety margin
    
    T = build_tensor()
    
    # Try ranks from low to high
    for r in range(9, SCHOOLBOOK + 1):
        if time.time() > deadline:
            break
        
        # Allocate time based on rank
        time_alloc = 10.0 if r <= 12 else 5.0
        if time.time() + time_alloc > deadline:
            time_alloc = deadline - time.time()
            if time_alloc < 1.0:
                break
        
        result = try_rank_structured(r, T, max_iters=5000, seed=42, time_limit=time_alloc)
        if result is not None:
            U, V, W = result
            # Verify
            if verify_decomposition(U, V, W, T):
                return {"U": [[float(x) for x in U[i]] for i in range(N)],
                        "V": [[float(x) for x in V[i]] for i in range(M)],
                        "W": [[float(x) for x in W[i]] for i in range(P)]}
    
    # Fallback: schoolbook decomposition
    rank = SCHOOLBOOK
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

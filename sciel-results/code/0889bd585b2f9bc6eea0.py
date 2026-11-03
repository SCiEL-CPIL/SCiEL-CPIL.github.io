"""Program for matrix multiplication 3,3,6 tensor decomposition.
Goal: find a rank-minimal exact decomposition (lower rank = better score)."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE_S = 110.0  # safety margin within 120s limit

def _verify(U, V, W, n=3, m=3, p=6, tol=1e-10):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n=3, m=3, p=6):
    """Build the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

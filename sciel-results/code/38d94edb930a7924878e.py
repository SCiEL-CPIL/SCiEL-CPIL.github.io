# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify decomposition using numpy for speed."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    result = np.einsum('ac,bc,dc->abd', U_np, V_np, W_np, optimize=True)
    
    return np.allclose(result, T, atol=1e-9), result, T

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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

def construct():
    """
    Matrix multiplication 3x3 by 3x6.
    Returns a valid tensor decomposition.
    """
    n, m, p = DIMENSIONS
    
    result = _schoolbook(n, m, p)
    ok, _, _ = _verify_decomposition(result["U"], result["V"], result["W"], n, m, p)
    if not ok:
        raise RuntimeError("Schoolbook verification failed")
    
    return result
# EVOLVE-BLOCK-END

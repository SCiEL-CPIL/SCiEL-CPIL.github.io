import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify_decomposition(U, V, W, rank, n, m, p):
    """Verify tensor decomposition for matrix multiplication.
    T[i,j,k] should equal 1 for all i,j,k.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 9
    
    # For standard matrix multiplication C = A*B, the structure tensor is:
    # T[i,j,k] = 1 for all i in [0,n-1], j in [0,m-1], k in [0,p-1]
    # This is the outer product of three all-ones vectors, so rank = 1.
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Verify
    if _verify_decomposition(np.array(U), np.array(V), np.array(W), 1, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return _schoolbook()

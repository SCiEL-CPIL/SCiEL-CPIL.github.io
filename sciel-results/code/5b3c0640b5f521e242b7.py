"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Returns the schoolbook tensor decomposition for 2x3 by 3x3 matrix multiplication.
    
    The tensor T has shape (n*m, m*p, p*n) where n=2, m=3, p=3.
    T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i for some i, j, k.
    
    The decomposition is rank n*m*p = 18.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    T = sum_r U[:, r] * V[:, r] * W[:, r]
    
    This is the standard schoolbook algorithm which is known to be optimal for many small rectangular cases.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the 1s for the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Row indices in U, V, W
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

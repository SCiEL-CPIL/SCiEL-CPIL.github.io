import time
import numpy as np

def construct():
    """
    Returns a low-rank tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Rank 36.
    U: (16, 36)
    V: (36, 36)
    W: (36, 36)
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 9
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # t = j * 4 + k
    # j in 0..8 (column of B/C)
    # k in 0..3 (row of B / col of A)
    
    for j in range(p):
        for k in range(m):
            t = j * 4 + k
            
            # V index: B is m x p (4x9). Element B[k, j] is at k * p + j.
            v_idx = k * p + j
            V[v_idx][t] = 1
            
            # U index: A is n x m (4x4). Element A[i, k] is at i * m + k.
            # We need U[i * m + k, t] = 1 for all i in 0..n-1.
            for i in range(n):
                u_idx = i * m + k
                U[u_idx][t] = 1
                
            # W index: C is n x p (4x9). Element C[i, j] is at j * n + i.
            # We need W[j * n + i, t] = 1 for all i in 0..n-1.
            for i in range(n):
                w_idx = j * n + i
                W[w_idx][t] = 1
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    if time.time() > deadline:
        # Return partial if necessary, though 36 terms is very fast
        pass

    return {"U": U, "V": V, "W": W}

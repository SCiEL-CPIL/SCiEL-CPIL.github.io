import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 110.0

def _time_left():
    return _DEADLINE - time.time()

def construct():
    n, m, p = 2, 4, 8
    A_dim = n * m   # 8
    B_dim = m * p   # 32
    C_dim = p * n   # 16
    
    # Decompose into two 1x4 x 4x8 row-vector multiplications.
    # For each row i of A (i=0,1) and each column j of A (j=0..3):
    #   C[k, i] += A[i, j] * B[j, k] for all k=0..7
    # This gives 2*4 = 8 rank-1 terms.
    
    rank = 8
    U = np.zeros((A_dim, rank), dtype=float)
    V = np.zeros((B_dim, rank), dtype=float)
    W = np.zeros((C_dim, rank), dtype=float)
    
    col = 0
    for i in range(n):       # row of A / column of C
        for j in range(m):   # col of A / row of B
            # U: A[i,j] -> flat index i*m + j
            U[i * m + j, col] = 1.0
            # V: B[j,k] for all k -> flat index j*p + k
            for k in range(p):
                V[j * p + k, col] = 1.0
            # W: C[k,i] for all k -> flat index k*n + i
            for k in range(p):
                W[k * n + i, col] = 1.0
            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

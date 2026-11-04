import time
import numpy as np

_DEADLINE = time.time() + 110  # safety margin before 120s evaluator kill (with 10s slack)

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Returns a rank-144 schoolbook tensor decomposition for 4x4 by 4x9 matrix multiplication.
    
    U: (n*m, rank) = (16, 144)
    V: (m*p, rank) = (36, 144)
    W: (p*n, rank) = (36, 144)
    
    The tensor T[i,j,k] = delta_{i,j+k} for the multiplication C = A * B
    where A is n x m, B is m x p, C is n x p.
    W index convention: W[k*n + i, r]
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    rank = n * m * p  # 4*4*9 = 144
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):       # row of A / row of C
        for j in range(m):   # col of A / row of B
            for k in range(p):  # col of B / col of C
                # A index: i*m + j  (0..15)
                a_idx = i * m + j
                # B index: j*p + k  (0..35)
                b_idx = j * p + k
                # W index: k*n + i  (0..35) per convention W[k*n+i, r]
                w_idx = k * n + i
                
                U[a_idx, idx] = 1.0
                V[b_idx, idx] = 1.0
                W[w_idx, idx] = 1.0
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

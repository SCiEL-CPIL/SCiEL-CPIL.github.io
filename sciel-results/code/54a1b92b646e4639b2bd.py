import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 24 decomposition
    # Structure: 
    # Split n=2 into 2 blocks of n=1.
    # Split p=8 into 4 blocks of p=2.
    # For each (n_block, p_block), we have m=3 terms.
    # Total rank = 2 * 4 * 3 = 24.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Mapping rank index r to (i, b, j)
    # i in [0, 1]
    # b in [0, 3] (block index for p, each block has 2 k's)
    # j in [0, 2]
    
    for r in range(rank):
        i = r // 12
        rem = r % 12
        b = rem // 3
        j = rem % 3
        
        # U index: i*m + j
        U[i * m + j][r] = 1
        
        # V indices: j*p + k for k in {2b, 2b+1}
        k1 = 2 * b
        k2 = 2 * b + 1
        V[j * p + k1][r] = 1
        V[j * p + k2][r] = 1
        
        # W indices: k*n + i for k in {2b, 2b+1}
        W[k1 * n + i][r] = 1
        W[k2 * n + i][r] = 1
        
    return {"U": U, "V": V, "W": W}

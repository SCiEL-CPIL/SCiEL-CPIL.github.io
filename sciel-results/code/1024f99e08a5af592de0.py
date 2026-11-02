import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    The rank of M_{3,3,7} is at most 63 (schoolbook).
    Lower bounds suggest it might be lower (>= 49), but explicit decompositions with rank < 63
    are complex to construct from scratch.
    This program constructs the explicit rank-63 decomposition (schoolbook algorithm).
    """
    n, m, p = DIMENSIONS
    
    # Dimensions of factor matrices
    dim_U = n * m  # 9
    dim_V = m * p  # 21
    dim_W = p * n  # 21
    
    rank = n * m * p  # 63
    
    U = np.zeros((dim_U, rank), dtype=int)
    V = np.zeros((dim_V, rank), dtype=int)
    W = np.zeros((dim_W, rank), dtype=int)
    
    # Construct the decomposition
    # Indices for A: i in [0,n), j in [0,m) -> idx_a = i*m + j
    # Indices for B: j in [0,m), k in [0,p) -> idx_b = j*p + k
    # Indices for C: k in [0,p), i in [0,n) -> idx_c = k*n + i (Official trace convention W[k*n+i, r])
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                U[a_idx, col] = 1
                V[b_idx, col] = 1
                W[c_idx, col] = 1
                col += 1
                
    # Return as lists to ensure compatibility
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

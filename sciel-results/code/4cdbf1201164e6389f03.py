import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # Rank 56 solution using 8 blocks of Strassen (rank 7)
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2 matrix multiplication
    # U vectors (for A)
    u_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    # V vectors (for B)
    v_vecs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    # W vectors (for C)
    w_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Indices mapping
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    for r_block in range(8):
        base_col = 7 * r_block
        # k indices for this block: 2*r_block, 2*r_block + 1
        k0 = 2 * r_block
        k1 = 2 * r_block + 1
        
        for s in range(7):
            col = base_col + s
            
            # Fill U
            # u_vecs[s] is [A00, A01, A10, A11]
            # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            U[0][col] = u_vecs[s][0]
            U[1][col] = u_vecs[s][1]
            U[2][col] = u_vecs[s][2]
            U[3][col] = u_vecs[s][3]
            
            # Fill V
            # v_vecs[s] is [B0k0, B0k1, B1k0, B1k1]
            # V indices:
            # 0: j=0, k=k0 -> 0*p + k0 = k0
            # 1: j=0, k=k1 -> 0*p + k1 = k1
            # 2: j=1, k=k0 -> 1*p + k0 = p + k0
            # 3: j=1, k=k1 -> 1*p + k1 = p + k1
            V[k0][col] = v_vecs[s][0]
            V[k1][col] = v_vecs[s][1]
            V[p + k0][col] = v_vecs[s][2]
            V[p + k1][col] = v_vecs[s][3]
            
            # Fill W
            # w_vecs[s] is [C0k0, C0k1, C1k0, C1k1]
            # W indices: k*n + i
            # 0: k=k0, i=0 -> k0*n + 0 = 2*k0
            # 1: k=k1, i=0 -> k1*n + 0 = 2*k1
            # 2: k=k0, i=1 -> k0*n + 1 = 2*k0 + 1
            # 3: k=k1, i=1 -> k1*n + 1 = 2*k1 + 1
            W[2*k0][col] = w_vecs[s][0]
            W[2*k1][col] = w_vecs[s][1]
            W[2*k0 + 1][col] = w_vecs[s][2]
            W[2*k1 + 1][col] = w_vecs[s][3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

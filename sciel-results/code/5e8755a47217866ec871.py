"""Matrix multiplication 2x3 by 3x8 tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Returns a rank-36 decomposition for 2x3 * 3x8 matrix multiplication.
    U: (6, 36)
    V: (24, 36)
    W: (16, 36)
    """
    n, m, p = 2, 3, 8
    
    # Rank for 2x3 * 3x4 is 18.
    # Total rank is 2 * 18 = 36.
    rank = 36
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to fill a 2x3 * 3x4 block
    def fill_block(offset, k_start):
        # offset: column offset in U, V, W
        # k_start: starting k index (0 or 4)
        
        # Part 1: j in {0, 1} (16 ranks)
        # k in {0, 1, 2, 3} relative to k_start
        
        for local_k in range(4):
            k = k_start + local_k
            
            # t1: A[i=0, j=0] * B[j=0, k] -> C[i=0, k]
            # a = 0*3+0=0, b = 0*8+k, c = k*2+0
            col = offset + local_k * 4 + 0
            U[0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + 0, col] = 1
            
            # t2: A[i=0, j=1] * B[j=1, k] -> C[i=0, k]
            # a = 0*3+1=1, b = 1*8+k, c = k*2+0
            col = offset + local_k * 4 + 1
            U[1, col] = 1
            V[1*p + k, col] = 1
            W[k*n + 0, col] = 1
            
            # t3: A[i=1, j=0] * B[j=0, k] -> C[i=1, k]
            # a = 1*3+0=3, b = 0*8+k, c = k*2+1
            col = offset + local_k * 4 + 2
            U[3, col] = 1
            V[0*p + k, col] = 1
            W[k*n + 1, col] = 1
            
            # t4: A[i=1, j=1] * B[j=1, k] -> C[i=1, k]
            # a = 1*3+1=4, b = 1*8+k, c = k*2+1
            col = offset + local_k * 4 + 3
            U[4, col] = 1
            V[1*p + k, col] = 1
            W[k*n + 1, col] = 1
            
        # Part 2: j = 2 (2 ranks)
        # d1: i=0
        col_d1 = offset + 16
        U[2, col_d1] = 1  # i=0, j=2 -> a=2
        # V for j=2, all k
        for local_k in range(4):
            k = k_start + local_k
            V[2*p + k, col_d1] = 1
        # W for i=0, all k
        for local_k in range(4):
            k = k_start + local_k
            W[k*n + 0, col_d1] = 1
            W[k*n + 1, col_d1] = 0 # i=1 is 0 for d1
            
        # d2: i=1
        col_d2 = offset + 17
        U[5, col_d2] = 1  # i=1, j=2 -> a=5
        # V for j=2, all k
        for local_k in range(4):
            k = k_start + local_k
            V[2*p + k, col_d2] = 1
        # W for i=1, all k
        for local_k in range(4):
            k = k_start + local_k
            W[k*n + 0, col_d2] = 0
            W[k*n + 1, col_d2] = 1

    # Fill Block 1 (k=0..3)
    fill_block(0, 0)
    
    # Fill Block 2 (k=4..7)
    fill_block(18, 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

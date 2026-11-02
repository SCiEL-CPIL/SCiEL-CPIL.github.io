import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # Rank 56 solution: 8 blocks of Strassen (rank 7 each)
    # The (2,2,16) multiplication tensor is a direct sum of 8 (2,2,2) tensors
    # Tensor rank of direct sum = sum of ranks = 8*7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 56)
    V = [[0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # shape (32, 56)
    
    # Strassen vectors for 2x2 matrix multiplication
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    u_vecs = [
        [1, 0, 0, 1],    # M1: A00+A11
        [0, 0, 1, 1],    # M2: A10+A11
        [1, 0, 0, 0],    # M3: A00
        [0, 0, 0, 1],    # M4: A11
        [1, 1, 0, 0],    # M5: A00+A01
        [-1, 0, 1, 0],   # M6: -A00+A10
        [0, 1, 0, -1]    # M7: A01-A11
    ]
    v_vecs = [
        [1, 0, 0, 1],    # M1: B00+B11
        [1, 0, 0, 0],    # M2: B00
        [0, 1, 0, -1],   # M3: B01-B11
        [-1, 0, 1, 0],   # M4: -B00+B10
        [0, 0, 0, 1],    # M5: B11
        [1, 1, 0, 0],    # M6: B00+B01
        [0, 0, 1, 1]     # M7: B10+B11
    ]
    # W vectors: local 0->C[0,0], 1->C[0,1], 2->C[1,0], 3->C[1,1]
    w_vecs = [
        [1, 0, 0, 1],    # M1: C00+C11
        [0, 0, 1, -1],   # M2: C10-C11
        [0, 1, 0, 1],    # M3: C01+C11
        [1, 0, 1, 0],    # M4: C00+C10
        [-1, 1, 0, 0],   # M5: -C00+C01
        [0, 0, 0, 1],    # M6: C11
        [1, 0, 0, 0]     # M7: C00
    ]
    
    # For each of the 8 blocks (r=0..7), handle B columns 2r and 2r+1
    for r in range(8):
        base_col = 7 * r
        # V indices for this block:
        # local 0 -> B[0, 2r] -> V[0*p + 2r] = V[2r]
        # local 1 -> B[0, 2r+1] -> V[2r+1]
        # local 2 -> B[1, 2r] -> V[1*p + 2r] = V[16+2r]
        # local 3 -> B[1, 2r+1] -> V[16+2r+1]
        v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
        
        # W indices for this block:
        # local 0 -> C[0, 2r] -> W[2r*n + 0] = W[4r]
        # local 1 -> C[0, 2r+1] -> W[(2r+1)*n + 0] = W[4r+2]
        # local 2 -> C[1, 2r] -> W[2r*n + 1] = W[4r+1]
        # local 3 -> C[1, 2r+1] -> W[(2r+1)*n + 1] = W[4r+3]
        w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
        
        for s in range(7):
            col = base_col + s
            
            # Fill U (A side - same for all blocks)
            U[0][col] = u_vecs[s][0]
            U[1][col] = u_vecs[s][1]
            U[2][col] = u_vecs[s][2]
            U[3][col] = u_vecs[s][3]
            
            # Fill V (B side - block-specific)
            V[v_idx[0]][col] = v_vecs[s][0]
            V[v_idx[1]][col] = v_vecs[s][1]
            V[v_idx[2]][col] = v_vecs[s][2]
            V[v_idx[3]][col] = v_vecs[s][3]
            
            # Fill W (C side - block-specific)
            W[w_idx[0]][col] = w_vecs[s][0]
            W[w_idx[1]][col] = w_vecs[s][1]
            W[w_idx[2]][col] = w_vecs[s][2]
            W[w_idx[3]][col] = w_vecs[s][3]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

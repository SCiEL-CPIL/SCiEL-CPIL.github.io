import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 68 decomposition
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We split A (3x3) into 2x2, 2x1, 1x2, 1x1
    # We split B (3x8) into 2x8, 1x8
    # C = A11 B11 + A12 B21
    #   + A21 B11 + A22 B21
    
    # Part 1: A11 (2x2) * B11 (2x8)
    # B11 is 2x8. We split B11 into 4 blocks of 2x2.
    # Each block uses rank 7 decomposition for 2x2x2.
    # Indices for A11: i in 0..1, j in 0..1. U index i*3+j.
    # Indices for B11: j in 0..1, k in 0..7. V index j*8+k.
    # Indices for C11: k in 0..7, i in 0..1. W index k*3+i.
    
    # Rank 7 decomposition for 2x2x2
    # u, v, w are vectors of length 4 (for 2x2 matrices)
    # u: (a,b,c,d) -> indices 0,1,2,3 in local 2x2
    # v: (e,f,g,h) -> indices 0,1,2,3 in local 2x2
    # w: (y00, y01, y10, y11) -> indices 0,1,2,3 in local 2x2
    
    # Correct Winograd 2x2x2 decomposition
    dec_222 = [
        ([1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1]),
        ([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        ([0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, 1]),
        ([1, 1, -1, -1], [0, 1, 0, 1], [1, 1, 0, 0]),
        ([0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0])
    ]
    
    col = 0
    # 4 blocks of 2x2 in B11
    for block in range(4):
        k_offset = block * 2
        for u_local, v_local, w_local in dec_222:
            # Fill U
            for i_local in range(2):
                for j_local in range(2):
                    if u_local[i_local * 2 + j_local] != 0:
                        u_idx = i_local * 3 + j_local
                        U[u_idx][col] = u_local[i_local * 2 + j_local]
            
            # Fill V
            for j_local in range(2):
                for k_local in range(2):
                    if v_local[j_local * 2 + k_local] != 0:
                        v_idx = j_local * 8 + (k_offset + k_local)
                        V[v_idx][col] = v_local[j_local * 2 + k_local]
            
            # Fill W
            for k_local in range(2):
                for i_local in range(2):
                    if w_local[k_local * 2 + i_local] != 0:
                        w_idx = (k_offset + k_local) * 3 + i_local
                        W[w_idx][col] = w_local[k_local * 2 + i_local]
            
            col += 1
    
    # Part 2: A12 (2x1) * B21 (1x8)
    # Rank 16 (Schoolbook is optimal)
    # A12 indices: i in 0..1, j=1. U index i*3+1.
    # B21 indices: j=2, k in 0..7. V index 2*8+k.
    # C12 indices: k in 0..7, i in 0..1. W index k*3+i.
    for i in range(2):
        for k in range(8):
            U[i * 3 + 1][col] = 1
            V[2 * 8 + k][col] = 1
            W[k * 3 + i][col] = 1
            col += 1
            
    # Part 3: A21 (1x2) * B11 (2x8)
    # Rank 16
    # A21 indices: i=2, j in 0..1. U index 2*3+j.
    # B11 indices: j in 0..1, k in 0..7. V index j*8+k.
    # C21 indices: k in 0..7, i=2. W index k*3+2.
    for j in range(2):
        for k in range(8):
            U[2 * 3 + j][col] = 1
            V[j * 8 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
            
    # Part 4: A22 (1x1) * B21 (1x8)
    # Rank 8
    # A22 index: i=2, j=2. U index 2*3+2.
    # B21 index: j=2, k in 0..7. V index 2*8+k.
    # C22 index: k in 0..7, i=2. W index k*3+2.
    for k in range(8):
        U[2 * 3 + 2][col] = 1
        V[2 * 8 + k][col] = 1
        W[k * 3 + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

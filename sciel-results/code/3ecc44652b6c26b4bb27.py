import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 2x2 * 2x16 matrix multiplication.
    Uses Strassen decomposition for each of the 8 independent 2x2 blocks.
    Rank = 8 * 7 = 56.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # p is 16, which is 8 blocks of 2 columns.
    num_blocks = p // 2
    rank = num_blocks * 7
    
    # Allocate matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for A (u), B (v), C (w)
    # u, v are length 4 (indices 00, 01, 10, 11)
    # w is length 4 (indices C00, C01, C10, C11)
    
    u_vecs = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [0, 0, 1, -1], # M6: c-a
        [0, 1, 0, -1]  # M7: b-d
    ]
    
    v_vecs = [
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [0, 0, 1, -1], # M4: g-e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1]   # M7: g+h
    ]
    
    # w_vecs correspond to contributions to C00, C01, C10, C11
    # M1: C00 + C11
    # M2: C10 - C11
    # M3: C01 + C11
    # M4: C00 + C10
    # M5: -C00 + C01
    # M6: C11
    # M7: C00
    w_vecs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    for t in range(num_blocks):
        if time.time() > deadline:
            break
        for s in range(7):
            col = t * 7 + s
            
            # Update U
            for idx in range(4):
                val = u_vecs[s][idx]
                if val != 0:
                    i = idx // 2
                    j = idx % 2
                    U[i * m + j][col] += val
            
            # Update V
            for idx in range(4):
                val = v_vecs[s][idx]
                if val != 0:
                    j = idx // 2
                    k_prime = idx % 2
                    k = 2 * t + k_prime
                    V[j * p + k][col] += val
            
            # Update W
            # Local C indices: 0:C00, 1:C01, 2:C10, 3:C11
            # Global W indices:
            # C00: k=2t, i=0 -> 2t*n + 0
            # C01: k=2t+1, i=0 -> (2t+1)*n + 0
            # C10: k=2t, i=1 -> 2t*n + 1
            # C11: k=2t+1, i=1 -> (2t+1)*n + 1
            
            w_vals = w_vecs[s]
            if w_vals[0] != 0:
                W[2 * t * n + 0][col] += w_vals[0]
            if w_vals[1] != 0:
                W[(2 * t + 1) * n + 0][col] += w_vals[1]
            if w_vals[2] != 0:
                W[2 * t * n + 1][col] += w_vals[2]
            if w_vals[3] != 0:
                W[(2 * t + 1) * n + 1][col] += w_vals[3]
                
    return {"U": U, "V": V, "W": W}

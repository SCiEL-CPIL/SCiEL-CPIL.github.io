import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * (p // 2) for p=6, so 21
    rank = 21
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # u vectors (coefficients for A)
    u_vecs = [
        [1, 0, 0, 1],  # M1: A11 + A22
        [0, 0, 1, 1],  # M2: A21 + A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11 + A12
        [-1, 0, 1, 0], # M6: A21 - A11
        [0, 1, 0, -1]  # M7: A12 - A22
    ]
    
    # v vectors (coefficients for B)
    # B indices: 0: B11, 1: B12, 2: B21, 3: B22
    v_vecs = [
        [1, 0, 0, 1],  # M1: B11 + B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1], # M3: B12 - B22
        [0, 0, 1, -1], # M4: B21 - B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11 + B12
        [0, 0, 1, 1]   # M7: B21 + B22
    ]
    
    # w vectors (coefficients for C)
    # C indices (col-major): 0: C11, 1: C21, 2: C12, 3: C22
    w_vecs = [
        [1, 0, 0, 1],  # M1: C11 + C22
        [0, 1, 0, -1], # M2: C21 - C22
        [0, 0, 1, 1],  # M3: C12 + C22
        [1, 1, 0, 0],  # M4: C11 + C21
        [-1, 0, 1, 0], # M5: -C11 + C12
        [0, 0, 0, 1],  # M6: C22
        [1, 0, 0, 0]   # M7: C11
    ]
    
    # There are p/2 = 3 blocks
    num_blocks = p // 2
    block_size = 2
    
    for b in range(num_blocks):
        base_col = b * 7
        # Fill U
        for i in range(4):
            for r in range(7):
                U[i][base_col + r] = u_vecs[r][i]
                
        # Fill V
        # B columns for this block are b*2 and b*2+1
        # V index is j*p + k
        # j=0, k=b*2 -> 0*6 + b*2 = 2b
        # j=0, k=b*2+1 -> 0*6 + b*2+1 = 2b+1
        # j=1, k=b*2 -> 1*6 + b*2 = 6+2b
        # j=1, k=b*2+1 -> 1*6 + b*2+1 = 7+2b
        
        v_indices = [2*b, 2*b+1, 6+2*b, 7+2*b]
        for r in range(7):
            for local_idx, v_idx in enumerate(v_indices):
                V[v_idx][base_col + r] = v_vecs[r][local_idx]
                
        # Fill W
        # C columns for this block are b*2 and b*2+1
        # W index is k*n + i
        # k=b*2, i=0 -> 2b*2 + 0 = 4b
        # k=b*2, i=1 -> 2b*2 + 1 = 4b+1
        # k=b*2+1, i=0 -> (2b+1)*2 + 0 = 4b+2
        # k=b*2+1, i=1 -> (2b+1)*2 + 1 = 4b+3
        
        w_indices = [4*b, 4*b+1, 4*b+2, 4*b+3]
        for r in range(7):
            for local_idx, w_idx in enumerate(w_indices):
                W[w_idx][base_col + r] = w_vecs[r][local_idx]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

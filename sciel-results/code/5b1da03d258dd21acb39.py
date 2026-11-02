import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 42 derived from padding p=11 to p=12
    # and using Strassen's algorithm on six 2x2x2 blocks.
    
    # Strassen coefficients for 2x2x2 all-ones tensor
    # U, V, W are lists of 7 vectors of length 4
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # For 2x2x2: U[0..3], V[0..3], W[0..3]
    
    U_s = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    V_s = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    W_s = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Verify Strassen locally (optional, but good for sanity)
    # Sum over r of U[i,j,r] V[j,k,r] W[k,i,r] should be 1
    
    # Construct for p=12
    p_pad = 12
    rank = 42
    U_12 = [[0] * rank for _ in range(n * m)]
    V_12 = [[0] * rank for _ in range(m * p_pad)]
    W_12 = [[0] * rank for _ in range(p_pad * n)]
    
    num_blocks = p_pad // 2
    for s in range(num_blocks):
        for t in range(7):
            col = s * 7 + t
            # U is same for all blocks
            for i in range(n):
                for j in range(m):
                    U_12[i * m + j][col] = U_s[t][i * m + j]
            
            # V mapping: j, k_local -> j, 2s + k_local
            for j in range(m):
                for k_loc in range(2):
                    k_global = 2 * s + k_loc
                    idx_12 = j * p_pad + k_global
                    idx_s = j * 2 + k_loc
                    V_12[idx_12][col] = V_s[t][idx_s]
            
            # W mapping: k_local, i -> 2s + k_local, i
            for k_loc in range(2):
                for i in range(n):
                    k_global = 2 * s + k_loc
                    idx_12 = k_global * n + i
                    idx_s = k_loc * n + i
                    W_12[idx_12][col] = W_s[t][idx_s]
    
    # Extract for p=11
    U = U_12
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            idx_11 = j * p + k
            idx_12 = j * p_pad + k
            V[idx_11] = V_12[idx_12]
            
    for k in range(p):
        for i in range(n):
            idx_11 = k * n + i
            idx_12 = k * n + i
            W[idx_11] = W_12[idx_12]
            
    return {"U": U, "V": V, "W": W}

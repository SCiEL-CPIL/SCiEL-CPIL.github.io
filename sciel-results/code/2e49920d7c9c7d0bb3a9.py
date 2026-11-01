"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Laderman's algorithm (Kronecker product of Strassen) for rank 49
    n, m, p = DIMENSIONS
    
    # Strassen vectors for 2x2 matrices
    # u, v, w are length 4 vectors corresponding to indices 0,1,2,3
    # Index mapping: 00->0, 01->1, 10->2, 11->3
    u_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    v_strassen = [
        [1, 1, 0, 0], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k1 in range(7):
        for k2 in range(7):
            r = k1 * 7 + k2
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    i1 = i // 2
                    i0 = i % 2
                    j1 = j // 2
                    j0 = j % 2
                    
                    u1_idx = i1 * 2 + j1
                    u2_idx = i0 * 2 + j0
                    
                    U[i * m + j][r] = u_strassen[k1][u1_idx] * u_strassen[k2][u2_idx]
            
            # Fill V
            for j in range(m):
                for k in range(p):
                    j1 = j // 2
                    j0 = j % 2
                    k1_idx = k // 2
                    k0_idx = k % 2
                    
                    v1_idx = j1 * 2 + k1_idx
                    v2_idx = j0 * 2 + k0_idx
                    
                    V[j * p + k][r] = v_strassen[k1][v1_idx] * v_strassen[k2][v2_idx]
            
            # Fill W
            # W index is k*n + i
            for k in range(p):
                for i in range(n):
                    k1_idx = k // 2
                    k0_idx = k % 2
                    i1 = i // 2
                    i0 = i % 2
                    
                    w1_idx = k1_idx * 2 + i1
                    w2_idx = k0_idx * 2 + i0
                    
                    W[k * n + i][r] = w_strassen[k1][w1_idx] * w_strassen[k2][w2_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

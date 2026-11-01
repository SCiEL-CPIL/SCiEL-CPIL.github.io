"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    # Rank 28 decomposition using 4 parallel Strassen 2x2x2 multiplications
    
    # Local Strassen coefficients for 2x2x2
    # U_local: 7 vectors of size 4 (A entries: 00, 01, 10, 11)
    # V_local: 7 vectors of size 4 (B entries: 00, 01, 10, 11)
    # W_local: 7 vectors of size 4 (C entries: 00, 01, 10, 11)
    
    U_local = [
        [1, 0, 0, 1], # M1: (a11+a22)
        [0, 0, 1, 1], # M2: (a21+a22)
        [1, 0, 0, 0], # M3: a11
        [0, 0, 0, 1], # M4: a22
        [1, 1, 0, 0], # M5: (a11+a12)
        [0, 0, -1, 1],# M6: (a21-a11)
        [0, 1, 0, -1] # M7: (a12-a22)
    ]
    
    V_local = [
        [1, 0, 0, 1], # M1: (b11+b22)
        [1, 0, 0, 0], # M2: b11
        [0, 1, 0, -1],# M3: (b12-b22)
        [0, 0, 1, -1],# M4: (b21-b11)
        [0, 0, 0, 1], # M5: b22
        [1, 1, 0, 0], # M6: (b11+b12)
        [0, 0, 1, 1]  # M7: (b21+b22)
    ]
    
    W_local = [
        [1, 0, 0, 1], # M1: c11+1, c22+1
        [0, 0, 1, -1],# M2: c21+1, c22-1
        [0, 1, 0, 1], # M3: c12+1, c22+1
        [1, 0, 1, 0], # M4: c11+1, c21+1
        [-1, 1, 0, 0],# M5: c11-1, c12+1
        [0, 0, 0, 1], # M6: c22+1
        [1, 0, 0, 0]  # M7: c11+1
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 4 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 16 x 28
    
    for k in range(4): # 4 blocks
        for s in range(7): # 7 Strassen steps
            r = k * 7 + s
            
            # Fill U
            for i in range(4):
                U[i][r] = U_local[s][i]
                
            # Fill V
            # Block k of B corresponds to indices 4k to 4k+3
            for i in range(4):
                V[4*k + i][r] = V_local[s][i]
                
            # Fill W
            # Block k of C corresponds to indices 4k to 4k+3
            for i in range(4):
                W[4*k + i][r] = W_local[s][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen 2x2 factors
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i (k is col, i is row of C)
    
    U_2x2 = [
        [1, 0, 0, 1], # M0
        [0, 0, 1, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 0, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 1, 0, -1] # M6
    ]
    
    V_2x2 = [
        [1, 0, 0, 1], # M0
        [1, 0, 0, 0], # M1
        [0, 1, 0, -1],# M2
        [-1, 0, 1, 0],# M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, 1]  # M6
    ]
    
    W_2x2 = [
        [1, 0, 0, 1], # M0: C00+1, C11+1. W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11
        [0, 1, 0, -1],# M1: C10+1, C11-1
        [0, 0, 1, 1], # M2: C01+1, C11+1
        [1, 0, 0, 0], # M3: C00+1
        [0, 1, 0, 1], # M4: C10+1, C11+1
        [-1, 0, 1, 0],# M5: C00-1, C01+1
        [-1, 0, 0, 0]  # M6: C00-1
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            # Construct U
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            idx_u = (2 * i1 + i2) * 4 + (2 * j1 + j2)
                            val = U_2x2[r1][i1 * 2 + j1] * U_2x2[r2][i2 * 2 + j2]
                            if val != 0:
                                U[idx_u][col] = val
                            
            # Construct V
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            idx_v = (2 * j1 + j2) * 4 + (2 * k1 + k2)
                            val = V_2x2[r1][j1 * 2 + k1] * V_2x2[r2][j2 * 2 + k2]
                            if val != 0:
                                V[idx_v][col] = val

            # Construct W
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            idx_w = (2 * k1 + k2) * 4 + (2 * i1 + i2)
                            val = W_2x2[r1][k1 * 2 + i1] * W_2x2[r2][k2 * 2 + i2]
                            if val != 0:
                                W[idx_w][col] = val
                            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

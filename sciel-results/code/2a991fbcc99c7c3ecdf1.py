"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrices
    # U2[r][i*2+j], V2[r][j*2+k], W2[r][k*2+i]
    U2 = [
        [0, 0, 0, 1], # M1: a00, a11 -> wait, U2[0] should be 1 at 0,0 and 1,1. Index 0 is 0,0. Index 3 is 1,1.
        # Correction: U2[0] = [1, 0, 0, 1]
    ]
    # Let's define them properly as lists of length 4
    U2 = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2: a10, a11
        [1, 0, 0, 0], # M3: a00
        [0, 0, 0, 1], # M4: a11
        [1, 1, 0, 0], # M5: a00, a01
        [-1, 0, 1, 0], # M6: a10 - a00
        [0, 1, 0, -1]  # M7: a01 - a11
    ]
    
    V2 = [
        [1, 0, 0, 1], # M1: b00, b11
        [1, 0, 0, 0], # M2: b00
        [0, 1, 0, -1], # M3: b01 - b11
        [-1, 0, 1, 0], # M4: b10 - b00
        [0, 0, 0, 1], # M5: b11
        [1, 1, 0, 0], # M6: b00 + b01
        [0, 0, 1, 1]  # M7: b10 + b11
    ]
    
    W2 = [
        [1, 0, 0, 1], # M1: c00, c11
        [0, 1, 0, -1], # M2: c10, -c11
        [0, 0, 1, 1], # M3: c01, c11
        [1, 1, 0, 0], # M4: c00, c10
        [-1, 0, 1, 0], # M5: -c00, c01
        [0, 0, 0, 1], # M6: c11
        [1, 0, 1, 0]  # M7: c00, c01
    ]
    
    # Construct 4x4 factors
    # Indices for 4x4:
    # U index: i*4 + j = (2*i0 + i1)*4 + (2*j0 + j1) = 8*i0 + 4*i1 + 2*j0 + j1
    # V index: j*4 + k = (2*j0 + j1)*4 + (2*k0 + k1) = 8*j0 + 4*j1 + 2*k0 + k1
    # W index: k*4 + i = (2*k0 + k1)*4 + (2*i0 + i1) = 8*k0 + 4*k1 + 2*i0 + i1
    
    for r in range(7):
        for s in range(7):
            col = 7 * r + s
            
            # Fill U
            for i0 in range(2):
                for i1 in range(2):
                    for j0 in range(2):
                        for j1 in range(2):
                            idx_u = 8 * i0 + 4 * i1 + 2 * j0 + j1
                            # U2 index for outer part (r): 2*i0 + j0
                            # U2 index for inner part (s): 2*i1 + j1
                            val_u = U2[r][2 * i0 + j0] * U2[s][2 * i1 + j1]
                            U[idx_u][col] = val_u
            
            # Fill V
            for j0 in range(2):
                for j1 in range(2):
                    for k0 in range(2):
                        for k1 in range(2):
                            idx_v = 8 * j0 + 4 * j1 + 2 * k0 + k1
                            # V2 index for outer part (r): 2*j0 + k0
                            # V2 index for inner part (s): 2*j1 + k1
                            val_v = V2[r][2 * j0 + k0] * V2[s][2 * j1 + k1]
                            V[idx_v][col] = val_v
            
            # Fill W
            for k0 in range(2):
                for k1 in range(2):
                    for i0 in range(2):
                        for i1 in range(2):
                            idx_w = 8 * k0 + 4 * k1 + 2 * i0 + i1
                            # W2 index for outer part (r): 2*k0 + i0
                            # W2 index for inner part (s): 2*k1 + i1
                            val_w = W2[r][2 * k0 + i0] * W2[s][2 * k1 + i1]
                            W[idx_w][col] = val_w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

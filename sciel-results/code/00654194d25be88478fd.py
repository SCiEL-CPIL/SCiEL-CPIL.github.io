import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen rank-7 decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # Strassen factors for A (U)
    # M1: a00+a11, M2: a10+a11, M3: a00, M4: a11, M5: a00+a01, M6: a10-a00, M7: a01-a11
    U_block = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    
    # Strassen factors for B (V)
    # M1: b00+b11, M2: b00, M3: b10-b11, M4: b01-b00, M5: b11, M6: b00+b01, M7: b10+b11
    V_block = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 0, 1, 0, 1, 0],  # b01
        [0, 0, 1, 0, 0, 0, 1],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ]
    
    # Strassen factors for C (W)
    # C00 = M1 - M5 + M4 - M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    W_block = [
        [1, 0, 0, 1, -1, 0, -1], # C00
        [0, 0, 0, 0, 1, 0, 1],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, -1, 0]  # C11
    ]
    
    rank = 7 * 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(8):
        for r in range(7):
            col = block * 7 + r
            # U is shared
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][col] = U_block[i * 2 + j][r]
            
            # V is block-specific
            for j in range(2):
                for k in range(2):
                    V[j * 16 + block * 2 + k][col] = V_block[j * 2 + k][r]
            
            # W is block-specific
            for k in range(2):
                for i in range(2):
                    W[(block * 2 + k) * 2 + i][col] = W_block[k * 2 + i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

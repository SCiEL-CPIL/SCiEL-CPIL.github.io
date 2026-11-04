"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 Strassen decomposition
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2 multiplication
    # Indexing: 0:00, 1:01, 2:10, 3:11
    u2 = [
        [1, 0, 0, 1],  # m1
        [0, 0, 1, 1],  # m2
        [1, 0, 0, 0],  # m3
        [0, 0, 0, 1],  # m4
        [1, 1, 0, 0],  # m5
        [0, 0, 1, -1], # m6
        [0, 1, 0, -1]  # m7
    ]
    v2 = [
        [1, 0, 0, 1],  # m1
        [1, 0, 0, 0],  # m2
        [0, 1, 0, -1], # m3
        [0, 0, 1, -1], # m4
        [0, 0, 0, 1],  # m5
        [1, 1, 0, 0],  # m6
        [0, 0, 1, 1]   # m7
    ]
    w2 = [
        [1, 0, 0, 1],  # m1 -> C00, C11
        [0, 0, 1, 0],  # m2 -> C10
        [0, 1, 0, 0],  # m3 -> C01
        [1, 0, 1, 0],  # m4 -> C00, C10
        [0, 1, 0, -1], # m5 -> C01, -C11
        [0, 0, 0, 1],  # m6 -> C11
        [1, 0, 0, 1]   # m7 -> C00, C11
    ]
    
    # Block-level Strassen matrices (2x2)
    # U_block[t][I][M]
    U_block = [
        [[1, 0], [0, 1]], # t=1: A00+A11
        [[0, 0], [1, 1]], # t=2: A10+A11
        [[1, 0], [0, 0]], # t=3: A00
        [[0, 0], [0, 1]], # t=4: A11
        [[1, 1], [0, 0]], # t=5: A00+A01
        [[-1, 0], [1, 0]],# t=6: A10-A00
        [[0, -1], [0, 1]] # t=7: A01-A11
    ]
    
    # V_block[t][M][L]
    V_block = [
        [[1, 0], [0, 1]], # t=1: B00+B11
        [[1, 0], [0, 0]], # t=2: B00
        [[0, 1], [0, -1]],# t=3: B01-B11
        [[-1, 0], [1, 0]],# t=4: B10-B00
        [[0, 0], [0, 1]], # t=5: B11
        [[1, 1], [0, 0]], # t=6: B00+B01
        [[0, 0], [1, 1]]  # t=7: B10+B11
    ]
    
    # W_block[t][K][L]
    W_block = [
        [[1, 0], [0, 1]], # t=1: C00, C11
        [[0, 0], [1, -1]],# t=2: C10, -C11
        [[0, 1], [0, 1]], # t=3: C01, C11
        [[1, 0], [1, 0]], # t=4: C00, C10
        [[-1, 1], [0, -1]],# t=5: -C00, C01, -C11
        [[0, 0], [0, 1]], # t=6: C11
        [[1, 0], [0, 0]]  # t=7: C00
    ]
    
    col = 0
    for t in range(7):
        for s in range(7):
            # Fill U
            for I in range(2):
                for M in range(2):
                    coeff_block = U_block[t][I][M]
                    if coeff_block != 0:
                        for p in range(2):
                            for q in range(2):
                                coeff_scalar = u2[s][p * 2 + q]
                                if coeff_scalar != 0:
                                    r = I * 2 + p
                                    c = M * 2 + q
                                    idx = r * n + c
                                    U[idx][col] += coeff_block * coeff_scalar
                                    
            # Fill V
            for M in range(2):
                for L in range(2):
                    coeff_block = V_block[t][M][L]
                    if coeff_block != 0:
                        for p in range(2):
                            for q in range(2):
                                coeff_scalar = v2[s][p * 2 + q]
                                if coeff_scalar != 0:
                                    r = M * 2 + p
                                    c = L * 2 + q
                                    idx = r * p + c
                                    V[idx][col] += coeff_block * coeff_scalar
                                    
            # Fill W
            for K in range(2):
                for L in range(2):
                    coeff_block = W_block[t][K][L]
                    if coeff_block != 0:
                        for p in range(2):
                            for q in range(2):
                                coeff_scalar = w2[s][p * 2 + q]
                                if coeff_scalar != 0:
                                    r = K * 2 + p
                                    c = L * 2 + q
                                    # Prompt W index: k*n + i. Here k is col, i is row.
                                    # c is col, r is row.
                                    idx = c * n + r
                                    W[idx][col] += coeff_block * coeff_scalar
                                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block Strassen approach.
    # Split B into 7 blocks of size 2x2.
    # Each block is multiplied by A using Strassen (rank 7).
    # Total rank = 7 * 7 = 49.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)] # 4 x 49
    V = [[0] * rank for _ in range(m * p)] # 28 x 49
    W = [[0] * rank for _ in range(p * n)] # 28 x 49
    
    # Strassen coefficients for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)h
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)e
    # M6 = (c+d)(g-f)
    # M7 = (a-b)(e+f)
    # S = M1 + M4 - M5 + M7
    # T = M3 + M1 - M2 + M6
    # U = M1 - M4 + M2 - M7
    # V = M3 + M5 - M1 + M6
    
    # Indices for A (4): 0: a, 1: b, 2: c, 3: d
    # Indices for B block k (4): 4k: e, 4k+1: f, 4k+2: g, 4k+3: h
    # Indices for C block k (4): 4k: S, 4k+1: T, 4k+2: U, 4k+3: V
    
    strassen_u = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [0, 0, 1, 1], # M6: c+d
        [1, -1, 0, 0] # M7: a-b
    ]
    
    strassen_v = [
        [1, 0, 1, 0], # M1: e+h
        [0, 0, 0, 1], # M2: h
        [0, 1, 0, -1], # M3: f-h
        [0, 0, 1, -1], # M4: g-e
        [1, 0, 0, 0], # M5: e
        [0, 0, 1, -1], # M6: g-f
        [1, 1, 0, 0]  # M7: e+f
    ]
    
    strassen_w = [
        [1, 0, 0, 0],   # S: M1
        [1, 1, 0, 0],   # T: M1
        [1, 0, 1, 0],   # U: M1
        [0, 1, 0, 1],   # V: M3? No.
        # Let's map carefully.
        # S = M1 + M4 - M5 + M7
        # T = M3 + M1 - M2 + M6
        # U = M1 - M4 + M2 - M7
        # V = M3 + M5 - M1 + M6
    ]
    
    # Correct W mapping for Strassen
    # Output indices: 0:S, 1:T, 2:U, 3:V
    # M1 contributes to S(1), T(1), U(1), V(-1)
    # M2 contributes to T(-1), U(1)
    # M3 contributes to T(1), V(1)
    # M4 contributes to S(1), U(-1)
    # M5 contributes to S(-1), V(1)
    # M6 contributes to T(1), V(1)
    # M7 contributes to S(1), U(-1)
    
    strassen_w = [
        [1, 1, 1, -1], # M1
        [0, -1, 1, 0], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, -1, 0], # M4
        [-1, 0, 0, 1], # M5
        [0, 1, 0, 1],  # M6
        [1, 0, -1, 0]  # M7
    ]
    
    col = 0
    for block in range(7):
        for s in range(7):
            # U part
            for i in range(4):
                U[i][col] = strassen_u[s][i]
            
            # V part
            # B indices for this block: 4*block to 4*block+3
            for j in range(4):
                V[4 * block + j][col] = strassen_v[s][j]
            
            # W part
            # C indices for this block: 4*block to 4*block+3
            for k in range(4):
                W[4 * block + k][col] = strassen_w[s][k]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

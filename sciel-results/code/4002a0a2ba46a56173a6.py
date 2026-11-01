"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # A indices: 0:00, 1:01, 2:10, 3:11
    # U factors (linear forms of A) for 7 products
    u_factors = [
        [1, 0, 0, 1],  # P1: a+d
        [0, 0, 1, 1],  # P2: c+d
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: a+b
        [-1, 0, 1, 0], # P6: c-a
        [0, 1, 0, -1]  # P7: b-d
    ]
    
    # V factors (linear forms of B block) for 7 products
    # B block indices for block b:
    # e = B[0, 2b] -> idx 2b
    # f = B[0, 2b+1] -> idx 2b+1
    # g = B[1, 2b] -> idx 10+2b
    # h = B[1, 2b+1] -> idx 10+2b+1
    # v_factors are defined relative to e, f, g, h
    # P1: e+h
    # P2: e
    # P3: f-h
    # P4: g-e
    # P5: h
    # P6: e+f
    # P7: g+h
    
    # W factors (linear forms of C block) for 7 products
    # C block indices for block b:
    # C00 = C[0, 2b] -> idx 4b
    # C01 = C[0, 2b+1] -> idx 4b+2
    # C10 = C[1, 2b] -> idx 4b+1
    # C11 = C[1, 2b+1] -> idx 4b+3
    
    # Coefficients for W
    # P1: C00 + C11
    # P2: C10 - C11
    # P3: C01 + C11
    # P4: C00 + C10
    # P5: -C00 + C01
    # P6: C11
    # P7: C00
    
    for b in range(5):
        for s in range(7):
            col = b * 7 + s
            
            # Fill U
            for i in range(4):
                U[i][col] = u_factors[s][i]
                
            # Fill V
            # e=2b, f=2b+1, g=10+2b, h=10+2b+1
            e = 2 * b
            f = 2 * b + 1
            g = 10 + 2 * b
            h = 10 + 2 * b + 1
            
            if s == 0:
                V[e][col] = 1
                V[h][col] = 1
            elif s == 1:
                V[e][col] = 1
            elif s == 2:
                V[f][col] = 1
                V[h][col] = -1
            elif s == 3:
                V[g][col] = 1
                V[e][col] = -1
            elif s == 4:
                V[h][col] = 1
            elif s == 5:
                V[e][col] = 1
                V[f][col] = 1
            elif s == 6:
                V[g][col] = 1
                V[h][col] = 1
                
            # Fill W
            c00 = 4 * b
            c01 = 4 * b + 2
            c10 = 4 * b + 1
            c11 = 4 * b + 3
            
            if s == 0:
                W[c00][col] = 1
                W[c11][col] = 1
            elif s == 1:
                W[c10][col] = 1
                W[c11][col] = -1
            elif s == 2:
                W[c01][col] = 1
                W[c11][col] = 1
            elif s == 3:
                W[c00][col] = 1
                W[c10][col] = 1
            elif s == 4:
                W[c00][col] = -1
                W[c01][col] = 1
            elif s == 5:
                W[c11][col] = 1
            elif s == 6:
                W[c00][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

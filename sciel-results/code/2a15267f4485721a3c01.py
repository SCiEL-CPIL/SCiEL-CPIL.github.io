"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # Block-Strassen: decompose into 2x2 blocks
    # A: 2x4 = 1 row-block x 2 col-blocks of 2x2
    # B: 4x8 = 2 row-blocks x 4 col-blocks of 2x2
    # C: 2x8 = 1 row-block x 4 col-blocks of 2x2
    # 8 sub-problems, each with 7 Strassen products -> rank 56
    
    total_subproblems = 2 * 2 * 4
    strassen_products = 7
    rank = total_subproblems * strassen_products  # 56
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * rank for _ in range(p * n)]   # 16 x 56
    
    # Strassen: (A_coeffs, B_coeffs, C_coeffs)
    # Local indices: (row, col) in {0,1}
    strassen_defs = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),       # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),                 # M2
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),                 # M3
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),                 # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (0,0):-1}),                 # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),                 # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),        # M7
    ]
    
    col = 0
    for ar in range(2):
        for br in range(2):
            for c in range(4):
                for prod_idx in range(7):
                    A_coeffs, B_coeffs, C_coeffs = strassen_defs[prod_idx]
                    
                    for (lr, ls), coeff in A_coeffs.items():
                        u_idx = lr * m + (ar * 2 + ls)
                        U[u_idx][col] += coeff
                    
                    for (lr, ls), coeff in B_coeffs.items():
                        v_idx = (br * 2 + lr) * p + (c * 2 + ls)
                        V[v_idx][col] += coeff
                    
                    for (lr, ls), coeff in C_coeffs.items():
                        w_idx = (c * 2 + ls) * n + lr
                        W[w_idx][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

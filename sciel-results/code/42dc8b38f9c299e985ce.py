"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # U_coeffs[s][a][b] is coefficient for A[a][b] in product s
    # V_coeffs[s][c][d] is coefficient for B[c][d] in product s
    # W_coeffs[s][e][f] is coefficient for C[e][f] from product s
    
    U_coeffs = [
        [[1, 0], [0, 1]], # P1
        [[0, 0], [1, 1]], # P2
        [[1, 0], [0, 0]], # P3
        [[0, 0], [0, 1]], # P4
        [[1, 1], [0, 0]], # P5
        [[-1, 0], [1, 0]], # P6
        [[0, 1], [0, -1]]  # P7
    ]
    
    V_coeffs = [
        [[1, 0], [0, 1]], # P1
        [[1, 0], [0, 0]], # P2
        [[0, 1], [0, -1]], # P3
        [[-1, 0], [1, 0]], # P4
        [[0, 0], [0, 1]], # P5
        [[1, 1], [0, 0]], # P6
        [[0, 0], [1, 1]]  # P7
    ]
    
    W_coeffs = [
        [[1, 0], [0, 1]], # P1
        [[0, 0], [1, 0]], # P2
        [[0, 1], [0, 0]], # P3
        [[1, 0], [1, 0]], # P4
        [[-1, 1], [0, 0]], # P5
        [[0, 0], [0, 1]], # P6
        [[1, 0], [0, 0]]  # P7
    ]

    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    # U: A[I, J] block
                    for a in range(2):
                        for b in range(2):
                            coeff = U_coeffs[s][a][b]
                            if coeff != 0:
                                row_u = (2 * I + a) * 6 + (2 * J + b)
                                U[row_u][col] = coeff
                            
                    # V: B[J, K] block
                    for c in range(2):
                        for d in range(2):
                            coeff = V_coeffs[s][c][d]
                            if coeff != 0:
                                row_v = (2 * J + c) * 6 + (2 * K + d)
                                V[row_v][col] = coeff
                                
                    # W: C[I, K] block
                    for e in range(2):
                        for f in range(2):
                            coeff = W_coeffs[s][e][f]
                            if coeff != 0:
                                # W index is k * n + i
                                # k = 2 * K + f
                                # i = 2 * I + e
                                row_w = (2 * K + f) * 6 + (2 * I + e)
                                W[row_w][col] = coeff
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

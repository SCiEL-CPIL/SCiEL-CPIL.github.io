"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using 3x3 blocks of 2x2 Strassen
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Coefficients for a11, a12, a21, a22
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Coefficients for b11, b12, b21, b22
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Coefficients for c11, c12, c21, c22
    C_coeffs = [
        [1, 0, 0, 0], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 0], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 1]  # M7
    ]
    
    column = 0
    for i1 in range(3):
        for j1 in range(3):
            for k1 in range(3):
                for t in range(7):
                    # Update U
                    for i2 in range(2):
                        for k2 in range(2):
                            idx_a = (2*i1 + i2) * m + (2*k1 + k2)
                            U[idx_a][column] += A_coeffs[t][i2 * 2 + k2]
                    
                    # Update V
                    for k2 in range(2):
                        for j2 in range(2):
                            idx_b = (2*k1 + k2) * p + (2*j1 + j2)
                            V[idx_b][column] += B_coeffs[t][k2 * 2 + j2]
                    
                    # Update W
                    for i2 in range(2):
                        for j2 in range(2):
                            idx_c = (2*j1 + j2) * n + (2*i1 + i2)
                            W[idx_c][column] += C_coeffs[t][i2 * 2 + j2]
                    
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen block decomposition for 6x6 using 2x2 blocks
    # 3x3 blocks of 2x2 matrices
    # Rank = 3 * 3 * 3 * 7 = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # x = [[x00, x01], [x10, x11]], y = [[y00, y01], [y10, y11]]
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)y00
    # M3 = x00(y10 - y11)
    # M4 = x11(y01 - y00)
    # M5 = (x00 + x01)y11
    # M6 = (x10 - x00)(y00 + y01)
    # M7 = (x01 - x11)(y10 + y11)
    
    # alpha[s] = coefficients for x (row-major: x00, x01, x10, x11)
    # beta[s] = coefficients for y (row-major: y00, y01, y10, y11)
    # gamma[s] = coefficients for z (row-major: z00, z01, z10, z11)
    
    alphas = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, -1, 0]  # M7
    ]
    
    betas = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 1, -1], # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    gammas = [
        [1, 0, 0, 1],  # M1: z00 += 1, z11 += 1
        [0, 0, 1, -1], # M2: z10 += 1, z11 += -1
        [0, 1, 0, 1],  # M3: z01 += 1, z11 += 1
        [1, 0, 1, 0],  # M4: z00 += 1, z10 += 1
        [-1, 1, 0, 0], # M5: z00 += -1, z01 += 1
        [0, 0, 0, 1],  # M6: z11 += 1
        [1, 0, 0, 0]   # M7: z00 += 1
    ]
    
    col = 0
    # Block indices I, J, K from 0 to 2
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    # U part: A[I, J] block
                    # Global indices: i = 2*I + u, j = 2*J + v
                    for u in range(2):
                        for v in range(2):
                            idx_A = (2*I + u) * m + (2*J + v)
                            U[idx_A][col] = alphas[s][u*2 + v]
                    
                    # V part: B[J, K] block
                    # Global indices: j = 2*J + v, k = 2*K + w
                    for v in range(2):
                        for w in range(2):
                            idx_B = (2*J + v) * p + (2*K + w)
                            V[idx_B][col] = betas[s][v*2 + w]
                    
                    # W part: C[I, K] block
                    # Global indices: k = 2*K + w, i = 2*I + u
                    # W index convention: k * n + i
                    for u_prime in range(2):
                        for w_prime in range(2):
                            idx_C = (2*K + w_prime) * n + (2*I + u_prime)
                            W[idx_C][col] = gammas[s][u_prime*2 + w_prime]
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

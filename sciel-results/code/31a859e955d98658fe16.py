import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use the Kronecker product of Strassen (2x2, rank 7) and schoolbook (3x3, rank 27)
    # Total rank = 7 * 27 = 189
    
    # Strassen factors for 2x2 matrix multiplication
    # Blocks: A00, A01, A10, A11 and B00, B01, B10, B11
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # alpha[s][i1][j1] is the coefficient of A[i1][j1] in X_s
    # beta[s][j1][k1] is the coefficient of B[j1][k1] in Y_s
    # gamma[s][i1][k1] is the coefficient of P_s in C[i1][k1]
    
    alpha = [
        [[1, 0], [0, 1]],  # P1
        [[0, 0], [1, 1]],  # P2
        [[1, 0], [0, 0]],  # P3
        [[0, 0], [0, 1]],  # P4
        [[1, 1], [0, 0]],  # P5
        [[-1, 0], [1, 0]], # P6
        [[0, 1], [0, -1]]  # P7
    ]
    
    beta = [
        [[1, 1], [0, 0]],  # P1
        [[1, 0], [0, 0]],  # P2
        [[0, 1], [0, -1]], # P3
        [[0, -1], [0, 1]], # P4
        [[0, 0], [0, 1]],  # P5
        [[1, 1], [0, 0]],  # P6
        [[0, 0], [1, 1]]   # P7
    ]
    
    gamma = [
        [[1, 0], [0, 1]],  # C00, C01, C10, C11
        [[0, 0], [1, 0]],  #
        [[0, 1], [0, 0]],  #
        [[1, 0], [0, 0]],  #
        [[-1, -1], [0, 0]],#
        [[0, 0], [0, 1]],  #
        [[1, 0], [0, 0]]   #
    ]
    
    # Correct gamma based on Strassen formulas:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    gamma = [
        [[1, 0], [0, 1]],  # P1
        [[0, 0], [1, -1]], # P2
        [[0, 1], [0, 1]],  # P3
        [[1, 0], [1, 0]],  # P4
        [[-1, 1], [0, 0]], # P5
        [[0, 0], [0, 1]],  # P6
        [[1, 0], [0, 0]]   # P7
    ]
    
    # Schoolbook factors for 3x3 matrix multiplication
    # U3[i*3+j, r], V3[j*3+k, r], W3[k*3+i, r]
    # r = i*9 + j*3 + k
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][r] = 1
                V3[j*3+k][r] = 1
                W3[k*3+i][r] = 1
                r += 1
                
    rank = 7 * 27
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    col = 0
    for s in range(7):
        for r3 in range(27):
            # Construct U, V, W for this term
            for i1 in range(2):
                for j1 in range(2):
                    a = alpha[s][i1][j1]
                    if a != 0:
                        for i2 in range(3):
                            for j2 in range(3):
                                u3 = U3[i2*3+j2][r3]
                                if u3 != 0:
                                    i = i1*3 + i2
                                    j = j1*3 + j2
                                    U[i*6+j][col] += a * u3
                                    
            for j1 in range(2):
                for k1 in range(2):
                    b = beta[s][j1][k1]
                    if b != 0:
                        for j2 in range(3):
                            for k2 in range(3):
                                v3 = V3[j2*3+k2][r3]
                                if v3 != 0:
                                    j = j1*3 + j2
                                    k = k1*3 + k2
                                    V[j*6+k][col] += b * v3
                                    
            for i1 in range(2):
                for k1 in range(2):
                    g = gamma[s][i1][k1]
                    if g != 0:
                        for i2 in range(3):
                            for k2 in range(3):
                                w3 = W3[k2*3+i2][r3]
                                if w3 != 0:
                                    k = k1*3 + k2
                                    i = i1*3 + i2
                                    W[k*6+i][col] += g * w3
                                    
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

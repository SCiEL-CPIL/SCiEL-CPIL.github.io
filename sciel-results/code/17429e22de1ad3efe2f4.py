import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen coefficients for 2x2 block multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # alpha[t][a][b] is coeff of A_ab in L_t
    # beta[t][c][d] is coeff of B_cd in R_t
    # gamma[t][e][f] is coeff of M_t in C_ef
    
    alpha = [
        [[1, 0], [0, 1]], # M1: A00 + A11
        [[0, 0], [1, 1]], # M2: A10 + A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00 + A01
        [[-1, 0], [1, 0]], # M6: A10 - A00
        [[0, 1], [0, -1]]  # M7: A01 - A11
    ]
    
    beta = [
        [[1, 0], [0, 1]], # M1: B00 + B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]], # M3: B01 - B11
        [[0, -1], [1, 0]], # M4: B10 - B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00 + B01
        [[0, 0], [1, 1]]  # M7: B10 + B11
    ]
    
    gamma = [
        [[1, 0], [0, 1]], # C00 += M1, C11 += M1
        [[0, 0], [1, 0]], # C10 += M2
        [[0, 1], [0, 0]], # C01 += M3
        [[1, 0], [0, 1]], # C00 += M4, C11 += M1 (wait, C11 = M1 - M2 + M3 + M6)
        [[0, -1], [0, 0]], # C01 -= M5
        [[0, 0], [0, 1]], # C11 += M6
        [[1, 0], [0, 0]]  # C00 += M7
    ]
    # Correct gamma:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    gamma = [
        [[1, 0], [0, 1]], # t=0 (M1): C00, C11
        [[0, 0], [1, -1]], # t=1 (M2): C10, C11(-)
        [[0, 1], [0, 1]], # t=2 (M3): C01, C11
        [[1, 0], [1, 0]], # t=3 (M4): C00, C10
        [[0, -1], [0, 0]], # t=4 (M5): C01(-)
        [[0, 0], [0, 1]], # t=5 (M6): C11
        [[1, 0], [0, 0]]  # t=6 (M7): C00
    ]
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(7):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for a in range(2):
                        for b in range(2):
                            # U index
                            I = a * 3 + i
                            J = b * 3 + j
                            U[I * 6 + J][col] = alpha[t][a][b]
                            
                            # V index
                            # J must match U's J? No, V's row is J.
                            # V row index is J (from U col).
                            # V col index is K.
                            # But V depends on b,d.
                            # In the sum, b is fixed by J.
                            # So V[J * 6 + K] = beta[t][b][d]
                            for d in range(2):
                                K = d * 3 + k
                                V[J * 6 + K][col] = beta[t][b][d]
                                
                                # W index
                                # W row is K, col is I.
                                # W[K * 6 + I] = gamma[t][a][d]
                                W[K * 6 + I][col] = gamma[t][a][d]
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen rank for 2x2 blocks of 3x3 is 7 * 27 = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 blocks
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # alpha[s][a][b] and beta[s][b][c]
    # a,b,c in {0,1}
    
    alpha = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    
    beta = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]], # M3
        [[0, -1], [0, 1]], # M4
        [[0, 0], [1, 0]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    
    for s in range(7):
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    # Iterate over possible block pairs
                    for a in range(2):
                        for b in range(2):
                            if alpha[s][a][b] == 0:
                                continue
                            # b' must equal b
                            b_prime = b
                            for c in range(2):
                                if beta[s][b_prime][c] == 0:
                                    continue
                                
                                coeff = alpha[s][a][b] * beta[s][b_prime][c]
                                if coeff == 0:
                                    continue
                                
                                # Global indices
                                i = 3 * a + u
                                j = 3 * b + v
                                k = 3 * c + w
                                
                                U[i * m + j][col] = coeff
                                V[j * p + k][col] = 1
                                W[k * n + i][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

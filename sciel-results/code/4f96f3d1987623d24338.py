import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 (Strassen) * 9 (Schoolbook 3x3) = 63
    rank = 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # alpha[s][r][c] for A blocks
    # beta[s][r][c] for B blocks
    # gamma[s][r][c] for C blocks
    # r, c in {0, 1}
    
    alpha = {}
    beta = {}
    gamma = {}
    
    # s=0: M0 = (A00 + A11)(B00 + B11)
    alpha[0] = {(0,0): 1, (1,1): 1}
    beta[0] = {(0,0): 1, (1,1): 1}
    
    # s=1: M1 = (A10 + A11)B00
    alpha[1] = {(1,0): 1, (1,1): 1}
    beta[1] = {(0,0): 1}
    
    # s=2: M2 = A00(B01 - B11)
    alpha[2] = {(0,0): 1}
    beta[2] = {(0,1): 1, (1,1): -1}
    
    # s=3: M3 = A11(B10 - B00)
    alpha[3] = {(1,1): 1}
    beta[3] = {(1,0): 1, (0,0): -1}
    
    # s=4: M4 = (A00 + A01)B11
    alpha[4] = {(0,0): 1, (0,1): 1}
    beta[4] = {(1,1): 1}
    
    # s=5: M5 = (A10 - A00)(B00 + B01)
    alpha[5] = {(1,0): 1, (0,0): -1}
    beta[5] = {(0,0): 1, (0,1): 1}
    
    # s=6: M6 = (A01 - A11)(B10 + B11)
    alpha[6] = {(0,1): 1, (1,1): -1}
    beta[6] = {(1,0): 1, (1,1): 1}
    
    # Gamma: C blocks
    # C00 = M0 + M3 - M4 + M6
    gamma[(0,0)] = {0: 1, 3: 1, 4: -1, 6: 1}
    # C01 = M2 + M4
    gamma[(0,1)] = {2: 1, 4: 1}
    # C10 = M1 + M3
    gamma[(1,0)] = {1: 1, 3: 1}
    # C11 = M0 - M1 + M2 + M5
    gamma[(1,1)] = {0: 1, 1: -1, 2: 1, 5: 1}

    for s in range(7):
        for u in range(3):
            for w in range(3):
                for v in range(3):
                    c = s * 27 + u * 9 + w * 3 + v
                    
                    # Fill U
                    for (r_a, c_a), coeff_a in alpha[s].items():
                        i = r_a * 3 + u
                        j = c_a * 3 + w
                        row_u = i * m + j
                        U[row_u][c] += coeff_a
                        
                    # Fill V
                    for (r_b, c_b), coeff_b in beta[s].items():
                        j = r_b * 3 + w
                        k = c_b * 3 + v
                        row_v = j * p + k
                        V[row_v][c] += coeff_b
                        
                    # Fill W
                    for (r_c, c_c), coeff_c in gamma.items():
                        if s in coeff_c:
                            i = r_c * 3 + u
                            k = c_c * 3 + v
                            row_w = k * n + i
                            W[row_w][c] += coeff_c[s]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

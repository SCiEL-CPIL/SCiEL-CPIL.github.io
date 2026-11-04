import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 49 + 8 = 57
    rank = 57
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps for 2x2x2
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Part 1: 4x4x4 part (cols 0-48)
    col_start_1 = 0
    
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            
            col = col_start_1 + s * 7 + s2
            
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
                    
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
                    
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # Part 2: 4x4x2 part (cols 49-56)
    # Rank 8 decomposition
    col_start_2 = 49
    
    for j in range(4):
        for k in range(2):
            col = col_start_2 + j * 2 + k
            
            # U: A_{ij} for all i
            for i in range(4):
                U[i * m + j][col] = 1
                
            # V: B_{j, 4+k}
            V[j * p + 4 + k][col] = 1
            
            # W: C_{i, 4+k} for all i
            for i in range(4):
                W[(4 + k) * n + i][col] = 1

    return {"U": U, "V": V, "W": W}

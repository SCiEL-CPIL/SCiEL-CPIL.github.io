import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    
    # Part 1: M_{1,4,4} for row 0 of A
    # Rank 16
    # Part 2: M_{2,4,4} for rows 1,2 of A
    # Rank 28 (4 * 7)
    
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 16 terms for A[0, :] * B
    for k in range(4):
        for j in range(4):
            U[0 * 4 + j][col] = 1
            V[j * 4 + k][col] = 1
            W[k * 3 + 0][col] = 1
            col += 1
            
    # Part 2: 28 terms for A[1:3, :] * B
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    blocks = [
        (1, 0, 0, 0, 1, 0),
        (1, 0, 0, 2, 1, 2),
        (1, 2, 2, 0, 1, 0),
        (1, 2, 2, 2, 1, 2),
    ]
    
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (x_coeffs, y_coeffs, z_coeffs) in strassen_terms:
            for (xi, yi), ux in x_coeffs.items():
                U[(ar + xi) * 4 + (ac + yi)][col] += ux
            for (yi, xi), vy in y_coeffs.items():
                V[(br + yi) * 4 + (bc + xi)][col] += vy
            for (zi, zj), wz in z_coeffs.items():
                W[(cc + zj) * 3 + (cr + zi)][col] += wz
            col += 1
            
    return {"U": U, "V": V, "W": W}

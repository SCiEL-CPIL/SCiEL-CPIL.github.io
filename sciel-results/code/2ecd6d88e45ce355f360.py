import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # a: A coeffs, b: B coeffs, c: C coeffs
    # M1
    a1 = [[1, 0], [0, 1]]
    b1 = [[1, 0], [0, 1]]
    c1 = [[1, 0], [0, 1]]
    # M2
    a2 = [[0, 0], [1, 1]]
    b2 = [[1, 0], [0, 0]]
    c2 = [[0, 0], [1, 0]]
    # M3
    a3 = [[1, 0], [0, 0]]
    b3 = [[0, 0], [1, -1]]
    c3 = [[0, 1], [0, 0]]
    # M4
    a4 = [[0, 0], [0, 1]]
    b4 = [[-1, 1], [0, 0]]
    c4 = [[1, 0], [0, 0]]
    # M5
    a5 = [[1, 1], [0, 0]]
    b5 = [[0, 0], [0, 1]]
    c5 = [[-1, 1], [0, 0]]
    # M6
    a6 = [[-1, 0], [1, 0]]
    b6 = [[1, 1], [0, 0]]
    c6 = [[0, 0], [0, 1]]
    # M7
    a7 = [[0, 1], [-1, 1]]
    b7 = [[0, 0], [1, 1]]
    c7 = [[1, 0], [0, 0]]
    
    strassen = [
        (a1, b1, c1),
        (a2, b2, c2),
        (a3, b3, c3),
        (a4, b4, c4),
        (a5, b5, c5),
        (a6, b6, c6),
        (a7, b7, c7)
    ]
    
    # 4 sub-blocks
    # (j_base, k_base)
    blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
    col = 0
    for j_base, k_base in blocks:
        for a, b, c in strassen:
            for i_l in range(2):
                for j_l in range(2):
                    if a[i_l][j_l] != 0:
                        i_g = i_l
                        j_g = j_base + j_l
                        U[i_g * 4 + j_g][col] = a[i_l][j_l]
            for j_l in range(2):
                for k_l in range(2):
                    if b[j_l][k_l] != 0:
                        j_g = j_base + j_l
                        k_g = k_base + k_l
                        V[j_g * 4 + k_g][col] = b[j_l][k_l]
            for k_l in range(2):
                for i_l in range(2):
                    if c[i_l][k_l] != 0:
                        k_g = k_base + k_l
                        i_g = i_l
                        W[k_g * 2 + i_g][col] = c[i_l][k_l]
            col += 1
            
    return {"U": U, "V": V, "W": W}

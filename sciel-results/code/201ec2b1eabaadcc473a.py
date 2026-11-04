import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A coeffs
    A_left = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    A_right = [
        [(0, 2, 1), (1, 3, 1)],
        [(1, 2, 1), (1, 3, 1)],
        [(0, 2, 1)],
        [(1, 3, 1)],
        [(0, 2, 1), (0, 3, 1)],
        [(1, 2, 1), (0, 2, -1)],
        [(0, 3, 1), (1, 3, -1)]
    ]
    
    # B coeffs
    B_top = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    B_bot = [
        [(2, 0, 1), (3, 1, 1)],
        [(2, 0, 1)],
        [(2, 1, 1), (3, 1, -1)],
        [(3, 0, 1), (2, 0, -1)],
        [(3, 1, 1)],
        [(2, 0, 1), (2, 1, 1)],
        [(3, 0, 1), (3, 1, 1)]
    ]
    
    # C coeffs
    C_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for t in range(4):
        base = t * 14
        # Left part of A
        for s in range(7):
            idx = base + s
            for i, j, c in A_left[s]:
                U[i * m + j][idx] += c
            for j, k_l, c in B_top[s]:
                k_g = t * 2 + k_l
                V[j * p + k_g][idx] += c
            for i, k_l, c in C_coeffs[s]:
                k_g = t * 2 + k_l
                W[k_g * n + i][idx] += c
                
        # Right part of A
        for s in range(7):
            idx = base + 7 + s
            for i, j, c in A_right[s]:
                U[i * m + j][idx] += c
            for j, k_l, c in B_bot[s]:
                k_g = t * 2 + k_l
                V[j * p + k_g][idx] += c
            for i, k_l, c in C_coeffs[s]:
                k_g = t * 2 + k_l
                W[k_g * n + i][idx] += c

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 sub-problems * 7 Strassen terms = 56
    rank = 56
    
    # Initialize matrices
    # U shape: (n*m, rank) = (8, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (16, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # Each term r has:
    # u_coeffs: list of (i, j, coeff)
    # v_coeffs: list of (j, k, coeff)
    # w_coeffs: list of (k, i, coeff)
    
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y10 - Y11)
    # M4 = X11(Y01 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    
    strassen_terms = [
        {
            'u': [(0, 0, 1), (1, 1, 1)],
            'v': [(0, 0, 1), (1, 1, 1)],
            'w': [(0, 0, 1), (1, 1, 1)] # Z00 += M1, Z11 += M1
        },
        {
            'u': [(1, 0, 1), (1, 1, 1)],
            'v': [(0, 0, 1)],
            'w': [(1, 0, 1), (1, 1, -1)] # Z10 += M2, Z11 -= M2
        },
        {
            'u': [(0, 0, 1)],
            'v': [(1, 0, 1), (1, 1, -1)],
            'w': [(0, 1, 1), (1, 1, 1)] # Z01 += M3, Z11 += M3
        },
        {
            'u': [(1, 1, 1)],
            'v': [(0, 1, 1), (0, 0, -1)],
            'w': [(0, 0, 1), (1, 0, 1)] # Z00 += M4, Z10 += M4
        },
        {
            'u': [(0, 0, 1), (0, 1, 1)],
            'v': [(1, 1, 1)],
            'w': [(0, 0, -1), (1, 1, 1)] # Z00 -= M5, Z11 += M5
        },
        {
            'u': [(1, 0, 1), (0, 0, -1)],
            'v': [(0, 0, 1), (0, 1, 1)],
            'w': [(1, 1, 1)] # Z11 += M6
        },
        {
            'u': [(0, 1, 1), (1, 1, -1)],
            'v': [(1, 0, 1), (1, 1, 1)],
            'w': [(0, 0, 1)] # Z00 += M7
        }
    ]
    
    # Sub-problem offsets
    # j_base in {0, 2}
    # k_base in {0, 2, 4, 6}
    
    col_idx = 0
    for j_base in [0, 2]:
        for k_base in [0, 2, 4, 6]:
            for r in range(7):
                term = strassen_terms[r]
                
                # Add U coefficients
                for i, j_local, coeff in term['u']:
                    j_global = j_base + j_local
                    # U index: i * m + j_global
                    U[i * m + j_global][col_idx] += coeff
                    
                # Add V coefficients
                for j_local, k_local, coeff in term['v']:
                    j_global = j_base + j_local
                    k_global = k_base + k_local
                    # V index: j_global * p + k_global
                    V[j_global * p + k_global][col_idx] += coeff
                    
                # Add W coefficients
                for k_local, i, coeff in term['w']:
                    k_global = k_base + k_local
                    # W index: k_global * n + i
                    W[k_global * n + i][col_idx] += coeff
                    
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}

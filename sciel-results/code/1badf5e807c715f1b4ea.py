import time
import numpy as np

def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    Uses a two-level Strassen decomposition: outer 2x2 block Strassen 
    combined with inner 2x2 element Strassen.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients (0-indexed)
    # a_coeffs[s][i][j]: coefficient of A[i][j] in s-th Strassen product
    # b_coeffs[s][j][k]: coefficient of B[j][k] in s-th Strassen product  
    # c_coeffs[s][i][k]: coefficient of M_s in C[i][k]
    
    a_coeffs = [
        [[1, 0], [0, 1]],   # M1 = (A00 + A11)
        [[0, 0], [1, 1]],   # M2 = (A10 + A11)
        [[1, 0], [0, 0]],   # M3 = A00
        [[0, 0], [0, 1]],   # M4 = A11
        [[1, 1], [0, 0]],   # M5 = (A00 + A01)
        [[-1, 0], [1, 0]],  # M6 = (A10 - A00)
        [[0, 1], [0, -1]],  # M7 = (A01 - A11)
    ]
    
    b_coeffs = [
        [[1, 0], [0, 1]],   # (B00 + B11)
        [[1, 0], [0, 0]],   # B00
        [[0, 1], [0, -1]],  # (B01 - B11)
        [[-1, 0], [1, 0]],  # (B10 - B00)
        [[0, 0], [0, 1]],   # B11
        [[1, 1], [0, 0]],   # (B00 + B01)
        [[0, 0], [1, 1]],   # (B10 + B11)
    ]
    
    c_coeffs = [
        [[1, 0], [0, 1]],   # C00 += M1, C11 += M1
        [[0, 0], [1, -1]],  # C10 += M2, C11 -= M2
        [[0, 1], [0, 1]],   # C01 += M3, C11 += M3
        [[1, 0], [1, 0]],   # C00 += M4, C10 += M4
        [[-1, 1], [0, 0]],  # C00 -= M5, C01 += M5
        [[0, 0], [0, 1]],   # C11 += M6
        [[1, 0], [0, 0]],   # C00 += M7
    ]
    
    # Build the 49-term decomposition
    # For each outer Strassen term s (0..6) and inner Strassen term t (0..6):
    # The combined term uses:
    #   U: a_coeffs[s][S][T] * a_coeffs[t][s2][t2] for A[2S+s2][2T+t2]
    #   V: b_coeffs[s][T][K] * b_coeffs[t][t2][k2] for B[2T+t2][2K+k2]
    #   W: c_coeffs[s][S][K] * c_coeffs[t][s2][k2] for C[2S+s2][2K+k2]
    #     but W is indexed as W[k*n+i] = W[k*4+i] where i=row, k=col
    
    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t
            
            # Fill U: row index = i*4 + j where i=row, j=col of A
            for S in range(2):
                for T in range(2):
                    coeff_outer_a = a_coeffs[s][S][T]
                    if coeff_outer_a == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            coeff_inner_a = a_coeffs[t][s2][t2]
                            if coeff_inner_a == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j][c_idx] += coeff_outer_a * coeff_inner_a
            
            # Fill V: row index = j*4 + k where j=row, k=col of B
            for T in range(2):
                for K in range(2):
                    coeff_outer_b = b_coeffs[s][T][K]
                    if coeff_outer_b == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            coeff_inner_b = b_coeffs[t][t2][k2]
                            if coeff_inner_b == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k][c_idx] += coeff_outer_b * coeff_inner_b
            
            # Fill W: row index = k*4 + i where k=col, i=row of C (trace convention W[k*n+i])
            for S in range(2):
                for K in range(2):
                    coeff_outer_c = c_coeffs[s][S][K]
                    if coeff_outer_c == 0:
                        continue
                    for s2 in range(2):
                        for k2 in range(2):
                            coeff_inner_c = c_coeffs[t][s2][k2]
                            if coeff_inner_c == 0:
                                continue
                            i = 2 * S + s2
                            k = 2 * K + k2
                            W[k * 4 + i][c_idx] += coeff_outer_c * coeff_inner_c
    
    return {"U": U, "V": V, "W": W}

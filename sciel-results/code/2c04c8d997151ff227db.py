"""Tensor decomposition for 6x6 matrix multiplication using block-Strassen."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Rank 189 decomposition:
    # Partition 6x6 into 3x3 blocks of 2x2 matrices.
    # Use Strassen (Rank 7) for each 2x2 block multiplication.
    # Total Rank = 27 * 7 = 189.
    
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 189): U[i*m+k, r] for A[i,k]
    V = [[0] * rank for _ in range(m * p)]  # (36, 189): V[k*p+j, r] for B[k,j]
    W = [[0] * rank for _ in range(p * n)]  # (36, 189): W[j*n+i, r] for C[i,j]
    
    # Strassen 7-term decomposition for 2x2 matrix multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Strassen terms: (U_offsets, U_coeffs, V_offsets, V_coeffs)
    # Offsets: 0=A00, 1=A01, 2=A10, 3=A11
    strassen_terms = [
        (([0, 3], [1, 1]), ([0, 3], [1, 1])),
        (([2, 3], [1, 1]), ([0], [1])),
        (([0], [1]), ([1, 3], [1, -1])),
        (([3], [1]), ([2, 0], [1, -1])),
        (([0, 1], [1, 1]), ([3], [1])),
        (([2, 0], [1, -1]), ([0, 1], [1, 1])),
        (([1, 3], [1, -1]), ([2, 3], [1, 1])),
    ]
    
    # C output contributions: list of (c_offset, coeff) for each term
    # C offsets: 0=C00, 1=C01, 2=C10, 3=C11
    c_contributions = [
        [(0, 1), (3, 1)],   # M0
        [(2, 1), (3, -1)],  # M1
        [(1, 1), (3, 1)],   # M2
        [(0, 1), (2, 1)],   # M3
        [(0, -1), (1, 1)],  # M4
        [(3, 1)],           # M5
        [(0, 1)],           # M6
    ]
    
    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                for t in range(7):
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]
                    
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        gr = 2 * bi + r
                        gc = 2 * bk + c
                        U[gr * 6 + gc][col] += coeff
                    
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        gr = 2 * bk + r
                        gc = 2 * bj + c
                        V[gr * 6 + gc][col] += coeff
                    
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        gr = 2 * bi + r
                        gc = 2 * bj + c
                        W[gc * 6 + gr][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

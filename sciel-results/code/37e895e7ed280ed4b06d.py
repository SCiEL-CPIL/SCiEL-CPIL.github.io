"""Program for 2x4 x 4x8 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    """Return rank-56 Strassen-based decomposition for 2x4 by 4x8 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = 56
    
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    # b: block of A (2 blocks: rows 0-1, 2-3)
    # c: block of B (4 blocks: cols 0-1, 2-3, 4-5, 6-7)
    for b in range(2):
        for c in range(4):
            # Local indices for A (2x2 block)
            idx_a = 0 * m + 2 * b       # A[0, 2b]
            idx_b = 0 * m + 2 * b + 1   # A[0, 2b+1]
            idx_c_a = 1 * m + 2 * b     # A[1, 2b]
            idx_d = 1 * m + 2 * b + 1   # A[1, 2b+1]
            
            # Local indices for B (2x2 block)
            idx_e = 2 * b * p + 2 * c       # B[2b, 2c]
            idx_f = 2 * b * p + 2 * c + 1   # B[2b, 2c+1]
            idx_g = (2 * b + 1) * p + 2 * c     # B[2b+1, 2c]
            idx_h = (2 * b + 1) * p + 2 * c + 1 # B[2b+1, 2c+1]
            
            # Local indices for C (2x2 output block)
            idx_c11 = 2 * c * n + 0       # C[0, 2c]
            idx_c12 = (2 * c + 1) * n + 0 # C[0, 2c+1]
            idx_c21 = 2 * c * n + 1       # C[1, 2c]
            idx_c22 = (2 * c + 1) * n + 1 # C[1, 2c+1]
            
            # M1 = (a+d)(e+h)
            U[idx_a][col] += 1
            U[idx_d][col] += 1
            V[idx_e][col] += 1
            V[idx_h][col] += 1
            W[idx_c11][col] += 1
            W[idx_c22][col] += 1
            col += 1
            
            # M2 = (c+d)e
            U[idx_c_a][col] += 1
            U[idx_d][col] += 1
            V[idx_e][col] += 1
            W[idx_c21][col] += 1
            W[idx_c22][col] -= 1
            col += 1
            
            # M3 = a(f-h)
            U[idx_a][col] += 1
            V[idx_f][col] += 1
            V[idx_h][col] -= 1
            W[idx_c12][col] += 1
            W[idx_c22][col] += 1
            col += 1
            
            # M4 = d(g-e)
            U[idx_d][col] += 1
            V[idx_g][col] += 1
            V[idx_e][col] -= 1
            W[idx_c11][col] += 1
            W[idx_c21][col] += 1
            col += 1
            
            # M5 = (a+b)h
            U[idx_a][col] += 1
            U[idx_b][col] += 1
            V[idx_h][col] += 1
            W[idx_c11][col] -= 1
            W[idx_c12][col] += 1
            col += 1
            
            # M6 = (c-a)(e+f)
            U[idx_c_a][col] += 1
            U[idx_a][col] -= 1
            V[idx_e][col] += 1
            V[idx_f][col] += 1
            W[idx_c22][col] += 1
            col += 1
            
            # M7 = (b-d)(g+h)
            U[idx_b][col] += 1
            U[idx_d][col] -= 1
            V[idx_g][col] += 1
            V[idx_h][col] += 1
            W[idx_c11][col] += 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

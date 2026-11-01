import time
import sys

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 6 * 7 (Strassen) + 4 (Standard) = 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen blocks for first 12 columns (6 blocks of 2)
    for l in range(6):
        k1 = 2 * l
        k2 = 2 * l + 1
        
        # Indices in B (V)
        idx_e = 0 * p + k1  # B[0, k1]
        idx_f = 0 * p + k2  # B[0, k2]
        idx_g = 1 * p + k1  # B[1, k1]
        idx_h = 1 * p + k2  # B[1, k2]
        
        # Indices in C (W)
        idx_c00 = k1 * n + 0 # C[0, k1]
        idx_c01 = k2 * n + 0 # C[0, k2]
        idx_c10 = k1 * n + 1 # C[1, k1]
        idx_c11 = k2 * n + 1 # C[1, k2]
        
        # Strassen Multipliers
        # M1 = (a+d)(e+f)
        # M2 = (c+d)g
        # M3 = a(e-f)
        # M4 = d(g-h)
        # M5 = (a+b)h
        # M6 = (c-d)(g+h)
        # M7 = (b-c)(e+h)
        
        # A indices: 0:a, 1:b, 2:c, 3:d
        
        # Term 1: (a+d)(e+f)
        U[0][col_idx] = 1; U[3][col_idx] = 1
        V[idx_e][col_idx] = 1; V[idx_f][col_idx] = 1
        W[idx_c00][col_idx] = 1; W[idx_c01][col_idx] = 1; W[idx_c10][col_idx] = 1; W[idx_c11][col_idx] = -1
        col_idx += 1
        
        # Term 2: (c+d)g
        U[2][col_idx] = 1; U[3][col_idx] = 1
        V[idx_g][col_idx] = 1
        W[idx_c10][col_idx] = 1; W[idx_c11][col_idx] = -1
        col_idx += 1
        
        # Term 3: a(e-f)
        U[0][col_idx] = 1
        V[idx_e][col_idx] = 1; V[idx_f][col_idx] = -1
        W[idx_c01][col_idx] = 1; W[idx_c11][col_idx] = 1
        col_idx += 1
        
        # Term 4: d(g-h)
        U[3][col_idx] = 1
        V[idx_g][col_idx] = 1; V[idx_h][col_idx] = -1
        W[idx_c00][col_idx] = 1; W[idx_c10][col_idx] = 1
        col_idx += 1
        
        # Term 5: (a+b)h
        U[0][col_idx] = 1; U[1][col_idx] = 1
        V[idx_h][col_idx] = 1
        W[idx_c00][col_idx] = -1; W[idx_c01][col_idx] = 1
        col_idx += 1
        
        # Term 6: (c-d)(g+h)
        U[2][col_idx] = 1; U[3][col_idx] = -1
        V[idx_g][col_idx] = 1; V[idx_h][col_idx] = 1
        W[idx_c11][col_idx] = 1
        col_idx += 1
        
        # Term 7: (b-c)(e+h)
        U[1][col_idx] = 1; U[2][col_idx] = -1
        V[idx_e][col_idx] = 1; V[idx_h][col_idx] = 1
        W[idx_c00][col_idx] = 1; W[idx_c11][col_idx] = 1
        col_idx += 1

    # Last column k=12
    k = 12
    idx_b0 = 0 * p + k # B[0, k]
    idx_b1 = 1 * p + k # B[1, k]
    idx_c0 = k * n + 0 # C[0, k]
    idx_c1 = k * n + 1 # C[1, k]
    
    # C0 = a*b0 + b*b1
    # Term 1: a*b0
    U[0][col_idx] = 1
    V[idx_b0][col_idx] = 1
    W[idx_c0][col_idx] = 1
    col_idx += 1
    
    # Term 2: b*b1
    U[1][col_idx] = 1
    V[idx_b1][col_idx] = 1
    W[idx_c0][col_idx] = 1
    col_idx += 1
    
    # C1 = c*b0 + d*b1
    # Term 3: c*b0
    U[2][col_idx] = 1
    V[idx_b0][col_idx] = 1
    W[idx_c1][col_idx] = 1
    col_idx += 1
    
    # Term 4: d*b1
    U[3][col_idx] = 1
    V[idx_b1][col_idx] = 1
    W[idx_c1][col_idx] = 1
    col_idx += 1
    
    return {"U": U, "V": V, "W": W}

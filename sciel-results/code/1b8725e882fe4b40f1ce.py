"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms for remaining dimensions, targeting rank < 72.
The structure is a direct sum of disjoint variable sets, so the rank is
the sum of the ranks of the components: 28 (Strassen) + 16 + 16 + 8 = 68.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    # We construct the 68-rank decomposition directly into the matrices.
    # This avoids intermediate list structures.
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    # We don't know the exact rank yet, but we know it's 68.
    # We can pre-allocate or append. Pre-allocating is faster.
    rank = 68
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen(row_off_a, col_off_a, row_off_b, col_off_b):
        nonlocal col
        # a, b, c, d are A indices
        # e, f, g, h are B indices
        a_idx = row_off_a * m + col_off_a
        b_idx = row_off_a * m + col_off_a + 1
        c_idx = (row_off_a + 1) * m + col_off_a
        d_idx = (row_off_a + 1) * m + col_off_a + 1
        
        e_idx = row_off_b * p + col_off_b
        f_idx = row_off_b * p + col_off_b + 1
        g_idx = (row_off_b + 1) * p + col_off_b
        h_idx = (row_off_b + 1) * p + col_off_b + 1
        
        C00_idx = col_off_b * n + row_off_a
        C01_idx = (col_off_b + 1) * n + row_off_a
        C10_idx = col_off_b * n + row_off_a + 1
        C11_idx = (col_off_b + 1) * n + row_off_a + 1
        
        # m1 = (a+d)(e+h)
        U[a_idx, col] = 1
        U[d_idx, col] = 1
        V[e_idx, col] = 1
        V[h_idx, col] = 1
        W[C00_idx, col] = 1
        W[C11_idx, col] = 1
        col += 1
        
        # m2 = (c+d)*e
        U[c_idx, col] = 1
        U[d_idx, col] = 1
        V[e_idx, col] = 1
        W[C10_idx, col] = 1
        W[C11_idx, col] = -1
        col += 1
        
        # m3 = a*(f-h)
        U[a_idx, col] = 1
        V[f_idx, col] = 1
        V[h_idx, col] = -1
        W[C01_idx, col] = 1
        W[C11_idx, col] = 1
        col += 1
        
        # m4 = d*(g-e)
        U[d_idx, col] = 1
        V[g_idx, col] = 1
        V[e_idx, col] = -1
        W[C00_idx, col] = 1
        W[C10_idx, col] = 1
        col += 1
        
        # m5 = (a+b)*h
        U[a_idx, col] = 1
        U[b_idx, col] = 1
        V[h_idx, col] = 1
        W[C00_idx, col] = -1
        W[C01_idx, col] = 1
        col += 1
        
        # m6 = (c-a)*(e+f)
        U[c_idx, col] = 1
        U[a_idx, col] = -1
        V[e_idx, col] = 1
        V[f_idx, col] = 1
        W[C11_idx, col] = 1
        col += 1
        
        # m7 = (b-d)*(g+h)
        U[b_idx, col] = 1
        U[d_idx, col] = -1
        V[g_idx, col] = 1
        V[h_idx, col] = 1
        W[C00_idx, col] = 1
        col += 1

    def add_direct(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1

    # 4 Strassen blocks for A[0:2, 0:2] * B[0:2, 0:8]
    # B is split into 4 blocks of width 2
    for k_block in range(4):
        add_strassen(0, 0, 0, k_block * 2)
        
    # Direct terms for A[0:2, 2] * B[2, 0:8]
    for i in range(2):
        for k in range(8):
            add_direct(i, 2, k)
            
    # Direct terms for A[2, 0:2] * B[0:2, 0:8]
    for j in range(2):
        for k in range(8):
            add_direct(2, j, k)
            
    # Direct terms for A[2, 2] * B[2, 0:8]
    for k in range(8):
        add_direct(2, 2, k)
        
    # Verify col is 68
    if col != 68:
        # This should not happen, but if it does, we have a bug.
        # We can fall back to schoolbook, but that's rank 72.
        pass

    # Verification
    # T[i*m+j, j*p+k, k*n+i] = 1
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if s != 1:
                    # Fallback to schoolbook if verification fails
                    rank_fb = n * m * p
                    U_fb = np.zeros((n * m, rank_fb), dtype=int)
                    V_fb = np.zeros((m * p, rank_fb), dtype=int)
                    W_fb = np.zeros((p * n, rank_fb), dtype=int)
                    c = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U_fb[i2*m+j2, c] = 1
                                V_fb[j2*p+k2, c] = 1
                                W_fb[k2*n+i2, c] = 1
                                c += 1
                    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

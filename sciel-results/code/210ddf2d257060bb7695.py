"""Optimized tensor decomposition for 2x3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a triple (u_vec, v_vec, w_vec) of length-4 vectors
    representing the rank-1 tensor components.
    
    For C = A * B where A, B are 2x2:
    C[0,0] = M1 - M4 + M6 + M7
    C[0,1] = M5 + M4 - M8 - M10  (wait, let me use standard Strassen)
    
    Standard Strassen:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 - M4 + M6
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M7
    """
    # Each term: (u, v, w) where u is in R^4 (A indices: 00,01,10,11),
    # v is in R^4 (B indices: 00,01,10,11), w is in R^4 (C indices: 00,01,10,11)
    
    # M1 = (A00+A11)*(B00+B11)
    # u = [1, 0, 0, 1], v = [1, 0, 0, 1], w = [1, 0, 0, 1] (contributes to C00 and C11)
    # Actually w needs to capture: C00 += M1, C11 += M1
    # So w = [1, 0, 0, 1]
    
    terms = []
    
    # M1 = (A00+A11)*(B00+B11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]  # C00 += M1, C11 += M1
    terms.append((u1, v1, w1))
    
    # M2 = (A10+A11)*B00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]  # C10 += M2
    terms.append((u2, v2, w2))
    
    # M3 = A00*(B01-B11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, -1]  # C01 += M3, C11 -= M3
    terms.append((u3, v3, w3))
    
    # M4 = A11*(B10-B00)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [-1, 0, 1, 0]  # C00 -= M4, C10 += M4
    terms.append((u4, v4, w4))
    
    # M5 = (A00+A01)*B11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [0, 1, 0, 0]  # C01 += M5
    terms.append((u5, v5, w5))
    
    # M6 = (A10-A00)*(B00+B01)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [1, 0, 0, 0]  # C00 += M6
    terms.append((u6, v6, w6))
    
    # M7 = (A01-A11)*(B10+B11)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [0, 0, 0, 1]  # C11 += M7
    terms.append((u7, v7, w7))
    
    return terms

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Decompose: 2x3x8 = 2x1x8 + 2x2x8
    # 2x1x8: rank 16 (schoolbook, no savings)
    # 2x2x8: split into four 2x2x2 blocks, each using Strassen (rank 7)
    # Total rank: 16 + 4*7 = 16 + 28 = 44
    
    rank = 44
    
    # U: shape (n*m, rank) = (6, 44)
    #    Row index: i*m + j where i in [0,n), j in [0,m)
    # V: shape (m*p, rank) = (24, 44)
    #    Row index: j*p + k where j in [0,m), k in [0,p)
    # W: shape (p*n, rank) = (16, 44)
    #    Row index: k*n + i where k in [0,p), i in [0,n)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x1x8 (j=0 only)
    # A is 2x1, B is 1x8, C is 2x8
    # C[i,k] = A[i,0] * B[0,k]
    # For each i in [0,2), k in [0,8):
    for i in range(n):
        for k in range(p):
            # U row: i*m + 0 = i*3 + 0
            U[i * m + 0][col] = 1
            # V row: 0*p + k = k
            V[0 * p + k][col] = 1
            # W row: k*n + i
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: 2x2x8 (j=1,2)
    # A is 2x2 (rows i=0,1; cols j=1,2), B is 2x8 (rows j=1,2; cols k=0..7)
    # C is 2x8 (rows i=0,1; cols k=0..7)
    # Split k into four groups of 2: k_group in [0,4), k = k_group*2 + kk where kk in [0,2)
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for k_group in range(4):
        for (u, v, w) in strassen_terms:
            # u indexes A[i, j] for i in [0,2), j in [1,2]
            #   A indices: (i, j) -> u[i*2 + (j-1)]
            #   So u[0]=A[0,1], u[1]=A[0,2], u[2]=A[1,1], u[3]=A[1,2]
            # v indexes B[j, k] for j in [1,2], k in [k_group*2, k_group*2+2)
            #   B indices: (j, k) -> v[(j-1)*2 + (k - k_group*2)]
            #   So v[0]=B[1,k0], v[1]=B[1,k0+1], v[2]=B[2,k0], v[3]=B[2,k0+1]
            # w indexes C[i, k] for i in [0,2), k in [k_group*2, k_group*2+2)
            #   C indices: (i, k) -> w[i*2 + (k - k_group*2)]
            #   So w[0]=C[0,k0], w[1]=C[0,k0+1], w[2]=C[1,k0], w[3]=C[1,k0+1]
            
            for idx in range(4):
                val = u[idx]
                if val != 0:
                    i = idx // 2
                    j = idx % 2 + 1  # j=1 or 2
                    U[i * m + j][col] += val
            
            for idx in range(4):
                val = v[idx]
                if val != 0:
                    j = idx // 2 + 1  # j=1 or 2
                    kk = idx % 2
                    k = k_group * 2 + kk
                    V[j * p + k][col] += val
            
            for idx in range(4):
                val = w[idx]
                if val != 0:
                    i = idx // 2
                    kk = idx % 2
                    k = k_group * 2 + kk
                    W[k * n + i][col] += val
            
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

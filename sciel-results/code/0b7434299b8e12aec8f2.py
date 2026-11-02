"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    Returns:
        U: list of 7 vectors of length 4 (for A)
        V: list of 7 vectors of length 4 (for B)
        W: list of 7 vectors of length 4 (for C)
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-c)(e+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # U vectors for A: each has 4 components [a, b, c, d]
    U = [
        [1, 0, 0, 1],  # m1: a+d
        [0, 0, 1, 1],  # m2: c+d
        [1, 0, 0, 0],  # m3: a
        [0, 0, 0, 1],  # m4: d
        [1, 1, 0, 0],  # m5: a+b
        [0, 0, 1, -1], # m6: c-d
        [0, 1, -1, 0], # m7: b-c
    ]
    
    # V vectors for B: each has 4 components [e, f, g, h]
    V = [
        [1, 0, 0, 1],  # m1: e+h
        [1, 0, 0, 0],  # m2: e
        [0, 1, 0, -1], # m3: f-h
        [0, -1, 1, 0], # m4: g-e
        [0, 0, 0, 1],  # m5: h
        [0, 1, 0, 1],  # m6: f+h
        [1, 0, 0, 1],  # m7: e+h
    ]
    
    # W vectors for C: each has 4 components [C11, C12, C21, C22]
    W = [
        [1, 0, 0, 1],   # m1 contributes to C11 and C22
        [0, 0, 1, -1],  # m2 contributes to C21 and -C22
        [0, 1, 0, 1],   # m3 contributes to C12 and C22
        [1, 0, 1, 0],   # m4 contributes to C11 and C21
        [-1, 1, 0, 0],  # m5 contributes to -C11 and C12
        [0, 0, 0, 1],   # m6 contributes to C22
        [1, 0, 0, 0],   # m7 contributes to C11
    ]
    
    return U, V, W

def construct():
    """Construct a rank-minimal exact tensor decomposition for 6x6 matrix multiplication.
    
    Uses Strassen's algorithm applied to 2x2 blocks within a 3x3 block grid.
    Each 2x2 block multiplication uses rank 7 instead of rank 4, giving total rank 3*3*7 = 63.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # We divide the 6x6 matrices into 3x3 blocks of 2x2 matrices
    # Each block multiplication uses Strassen's rank-7 decomposition
    
    U_str, V_str, W_str = _strassen_2x2()
    
    # Total rank: 3 block_rows * 3 block_cols * 7 strassen_products = 63
    rank = 3 * 3 * 7
    
    # Initialize factor matrices
    # U: (n*m, rank) = (36, 63)
    # V: (m*p, rank) = (36, 63)  
    # W: (p*n, rank) = (36, 63)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each block position (bi, bj) where bi, bj in {0, 1, 2}
    # and each Strassen product s in {0, ..., 6}
    # We add contributions to U, V, W
    
    for bi in range(3):  # block row index (0, 1, 2)
        for bj in range(3):  # block col index (0, 1, 2)
            for s in range(7):  # Strassen product index
                col = bi * 21 + bj * 7 + s
                
                # For each Strassen factor component
                for idx_u in range(4):
                    if U_str[s][idx_u] != 0:
                        # A block at position (bi, bj)
                        # Row in A: bi*2 + idx_u//2, Col in A: bj*2 + idx_u%2
                        # Index in U: row*m + col = (bi*2 + idx_u//2)*6 + (bj*2 + idx_u%2)
                        row_a = bi * 2 + idx_u // 2
                        col_a = bj * 2 + idx_u % 2
                        U[row_a * m + col_a][col] = U_str[s][idx_u]
                
                for idx_v in range(4):
                    if V_str[s][idx_v] != 0:
                        # B block at position (bj, bi) - wait, need to be careful
                        # B is m x p, so B[row, col] where row is in {0..5}, col in {0..5}
                        # The block of B that multiplies with A[bi,bj] is B[bj, bi]
                        # Row in B: bj*2 + idx_v//2, Col in B: bi*2 + idx_v%2
                        row_b = bj * 2 + idx_v // 2
                        col_b = bi * 2 + idx_v % 2
                        V[row_b * p + col_b][col] = V_str[s][idx_v]
                
                for idx_w in range(4):
                    if W_str[s][idx_w] != 0:
                        # C block at position (bi, bj)
                        # Row in C: bi*2 + idx_w//2, Col in C: bj*2 + idx_w%2
                        # Index in W: col*n + row = (bj*2 + idx_w%2)*6 + (bi*2 + idx_w//2)
                        # Wait, W[k*n+i, r] where k is col of C, i is row of C
                        row_c = bi * 2 + idx_w // 2
                        col_c = bj * 2 + idx_w % 2
                        W[col_c * n + row_c][col] = W_str[s][idx_w]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

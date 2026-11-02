"""
Rank-minimal exact tensor decomposition for 2x2 * 2x12 matrix multiplication.

Strategy:
- View the 12 columns of B as 6 independent 2x2 blocks.
- For each block, use Strassen's 7-product algorithm to compute the 2x2 product.
- Total rank = 6 * 7 = 42.

Indexing conventions:
  U[i*m+j, r]: row = i*m+j for A[i][j], where i in 0..1, j in 0..1
  V[j*p+k, r]: row = j*p+k for B[j][k], where j in 0..1, k in 0..11
  W[k*n+i, r]: row = k*n+i for C[i][k], where k in 0..11, i in 0..1 (trace convention)
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    """
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices
    for the rank-42 Strassen-based decomposition of 2x2 * 2x12 matmul.
    
    Each of the 6 column-pairs of B is treated as an independent 2x2 block,
    and Strassen's algorithm is applied to each block.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strassen's 7 products for a 2x2 block:
    # Given A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For each product s (0..6), define:
    #   A_factors: list of (i, j, coeff) meaning coeff * A[i][j]
    #   B_factors: list of (j, k, coeff) meaning coeff * B[j][k]
    #   C_contribs: list of (i, k, coeff) meaning coeff * C[i][k]
    
    # Product 1: (a+d)(e+h)
    p1_A = [(0,0,1), (1,1,1)]
    p1_B = [(0,0,1), (1,1,1)]
    p1_C = [(0,0,1), (1,1,-1)]
    
    # Product 2: (c+d)e
    p2_A = [(1,0,1), (1,1,1)]
    p2_B = [(0,0,1)]
    p2_C = [(1,0,1), (1,1,-1)]
    
    # Product 3: a(f-h)
    p3_A = [(0,0,1)]
    p3_B = [(0,1,1), (1,1,-1)]
    p3_C = [(0,1,1), (1,1,1)]
    
    # Product 4: d(g-e)
    p4_A = [(1,1,1)]
    p4_B = [(1,0,1), (0,0,-1)]
    p4_C = [(0,0,1), (1,0,1)]
    
    # Product 5: (a+b)h
    p5_A = [(0,0,1), (0,1,1)]
    p5_B = [(1,1,1)]
    p5_C = [(0,0,-1), (0,1,1)]
    
    # Product 6: (c-a)(e+f)
    p6_A = [(1,0,1), (0,0,-1)]
    p6_B = [(0,0,1), (0,1,1)]
    p6_C = [(1,1,1)]
    
    # Product 7: (b-d)(g+h)
    p7_A = [(0,1,1), (1,1,-1)]
    p7_B = [(1,0,1), (1,1,1)]
    p7_C = [(0,0,1)]
    
    products = [
        (p1_A, p1_B, p1_C),
        (p2_A, p2_B, p2_C),
        (p3_A, p3_B, p3_C),
        (p4_A, p4_B, p4_C),
        (p5_A, p5_B, p5_C),
        (p6_A, p6_B, p6_C),
        (p7_A, p7_B, p7_C),
    ]
    
    num_blocks = p // 2  # 6 blocks of 2 columns each
    rank = num_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for block in range(num_blocks):
        base_col = 2 * block
        for s in range(7):
            col = block * 7 + s
            A_factors, B_factors, C_factors = products[s]
            
            for (i, j, coeff) in A_factors:
                row = i * m + j
                U[row, col] = coeff
            
            for (j, k, coeff) in B_factors:
                actual_k = base_col + k
                row = j * p + actual_k
                V[row, col] = coeff
            
            for (i, k, coeff) in C_factors:
                actual_k = base_col + k
                row = actual_k * n + i
                W[row, col] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

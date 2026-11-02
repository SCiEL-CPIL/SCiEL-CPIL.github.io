DIMENSIONS = [2, 2, 12]

import numpy as np

def construct():
    """
    Rank-42 decomposition for 2x2 * 2x12 matrix multiplication.
    Uses Strassen's algorithm on 6 independent 2x2x2 column-pair blocks.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j for A[i][j]
      V[j*p+k, r]: row = j*p+k for B[j][k]
      W[k*n+i, r]: row = k*n+i for C[i][k] (trace convention)
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2    # 6
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for 2x2 * 2x2
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B11-B01)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each entry: list of (index_in_4, coefficient) for U, V; and (output_pos, coeff) for W
    strassen_products = [
        # P1
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # P2
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        # P3
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        # P4
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        # P5
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # P6
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        # P7
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),
    ]
    
    # Output contributions: for each product, which C entries it contributes to and with what coefficient
    # C is 2x2: C00, C01, C10, C11
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    output_contribs = [
        [((0,0), 1), ((1,1), 1)],      # P1: C00 += 1, C11 += 1
        [((1,0), 1), ((1,1), -1)],     # P2: C10 += 1, C11 -= 1
        [((0,1), 1), ((1,1), 1)],      # P3: C01 += 1, C11 += 1
        [((0,0), 1), ((1,0), 1)],      # P4: C00 += 1, C10 += 1
        [((0,0), -1), ((0,1), 1)],     # P5: C00 -= 1, C01 += 1
        [((1,1), 1)],                   # P6: C11 += 1
        [((0,0), 1)],                   # P7: C00 += 1
    ]
    
    col = 0
    for t in range(num_pairs):
        base_col = 2 * t
        for s in range(7):
            for (ai, aj), coeff in strassen_products[s][0]:
                U[ai * m + aj, col] = coeff
            
            for (bj, bk_rel), coeff in strassen_products[s][1]:
                V[bj * p + base_col + bk_rel, col] = coeff
            
            for (ci, ck_rel), coeff in output_contribs[s]:
                W[(base_col + ck_rel) * n + ci, col] = coeff
            
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

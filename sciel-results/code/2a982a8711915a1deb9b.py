"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition of 6x6 matrix multiplication.
    
    Strategy: View 6x6 as 3x3 blocks of 2x2 matrices.
    - Block level: schoolbook 3x3 matrix mult (27 block multiplications)
    - 2x2 level: Strassen algorithm (rank 7 per 2x2 mult)
    - Total rank: 27 * 7 = 189
    
    C = A * B, where C[i][j] = sum_k A[i][k] * B[k][j]
    Block (bi, bk) of C = sum_{bj} Block(bi,bj) of A * Block(bj,bk) of B
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen 2x2 decomposition coefficients
    # For 2x2 matrices A, B, C = A*B:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1
        {
            'a': {(0,0): 1, (1,1): 1},
            'b': {(0,0): 1, (1,1): 1},
            'c': {(0,0): 1, (1,1): 1}
        },
        # M2
        {
            'a': {(1,0): 1, (1,1): 1},
            'b': {(0,0): 1},
            'c': {(1,0): 1, (1,1): -1}
        },
        # M3
        {
            'a': {(0,0): 1},
            'b': {(1,0): 1, (1,1): -1},
            'c': {(0,1): 1, (1,1): 1}
        },
        # M4
        {
            'a': {(1,1): 1},
            'b': {(0,1): 1, (0,0): -1},
            'c': {(0,0): 1, (1,0): 1}
        },
        # M5
        {
            'a': {(0,0): 1, (0,1): 1},
            'b': {(1,1): 1},
            'c': {(0,0): -1, (0,1): 1}
        },
        # M6
        {
            'a': {(1,0): 1, (0,0): -1},
            'b': {(0,0): 1, (0,1): 1},
            'c': {(1,1): 1}
        },
        # M7
        {
            'a': {(0,1): 1, (1,1): -1},
            'b': {(1,0): 1, (1,1): 1},
            'c': {(0,0): 1}
        },
    ]
    
    rank = 27 * 7  # 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    for bi in range(3):  # block row of A and C
        for bk in range(3):  # block col of B and C
            for bj in range(3):  # block col of A / block row of B
                for s in range(7):  # Strassen term
                    term = strassen_terms[s]
                    
                    # A block (bi, bj): 2x2
                    # A row = bi*2+di, A col = bj*2+dj
                    # U index = (bi*2+di)*m + (bj*2+dj)
                    for (di, dj), a_coeff in term['a'].items():
                        row_a = bi * 2 + di
                        col_a = bj * 2 + dj
                        U[row_a * m + col_a][col] = a_coeff
                    
                    # B block (bj, bk): 2x2
                    # B row = bj*2+di, B col = bk*2+dj
                    # V index = (bj*2+di)*p + (bk*2+dj)
                    for (di, dj), b_coeff in term['b'].items():
                        row_b = bj * 2 + di
                        col_b = bk * 2 + dj
                        V[row_b * p + col_b][col] = b_coeff
                    
                    # C block (bi, bk): 2x2
                    # C row = bi*2+di, C col = bk*2+dj
                    # W index = (bi*2+di)*n + (bk*2+dj)  [W[k*n+i] where k=row, i=col]
                    for (di, dj), c_coeff in term['c'].items():
                        row_c = bi * 2 + di
                        col_c = bk * 2 + dj
                        W[row_c * n + col_c][col] = c_coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Matrix multiplication 6x6x6 via Strassen (2x2 blocks of 3x3) with schoolbook 3x3.
Total rank: 7 * 27 = 189.
"""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size

    # Strassen decomposition at the 2x2 block level
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B00)
    # P4 = A11(B01-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # Each term: (a_blocks, b_blocks, c_blocks_with_signs)
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 (for A)
    #                0=B00, 1=B01, 2=B10, 3=B11 (for B)
    #                0=C00, 1=C01, 2=C10, 3=C11 (for C)
    
    strassen = [
        # P1 = (A00+A11)(B00+B11)
        {
            'a': [(0, 1), (3, 1)],
            'b': [(0, 1), (3, 1)],
            'c': [(0, 1), (3, 1)],  # C00+, C11+
        },
        # P2 = (A10+A11)B00
        {
            'a': [(2, 1), (3, 1)],
            'b': [(0, 1)],
            'c': [(2, 1), (3, -1)],  # C10+, C11-
        },
        # P3 = A00(B10-B00)
        {
            'a': [(0, 1)],
            'b': [(2, 1), (0, -1)],
            'c': [(1, 1), (3, 1)],  # C01+, C11+
        },
        # P4 = A11(B01-B10)
        {
            'a': [(3, 1)],
            'b': [(1, 1), (2, -1)],
            'c': [(0, 1), (2, 1)],  # C00+, C10+
        },
        # P5 = (A00+A01)B11
        {
            'a': [(0, 1), (1, 1)],
            'b': [(3, 1)],
            'c': [(0, -1), (1, 1)],  # C00-, C01+
        },
        # P6 = (A10-A00)(B00+B01)
        {
            'a': [(2, 1), (0, -1)],
            'b': [(0, 1), (1, 1)],
            'c': [(3, 1)],  # C11+
        },
        # P7 = (A01-A11)(B10+B11)
        {
            'a': [(1, 1), (3, -1)],
            'b': [(2, 1), (3, 1)],
            'c': [(0, 1)],  # C00+
        },
    ]

    # Schoolbook rank-27 for 3x3 matrix multiplication
    # For each (i, k, j) in range(3)^3:
    #   U[i*3+k] = 1 (for A)
    #   V[k*3+j] = 1 (for B)  
    #   W[j*3+i] = 1 (for C, using W[col*3+row] convention)
    
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for term in strassen:
        a_blocks = term['a']  # list of (block_idx, coeff)
        b_blocks = term['b']
        c_blocks = term['c']
        
        for i in range(bs):
            for k in range(bs):
                for j in range(bs):
                    # A part: sum over blocks
                    for blk_a, coeff_a in a_blocks:
                        row_a = (blk_a // 2) * bs  # 0 for blk 0,1; 1 for blk 2,3
                        col_a = (blk_a % 2) * bs
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx][col] += coeff_a
                    
                    # B part: sum over blocks
                    for blk_b, coeff_b in b_blocks:
                        row_b = (blk_b // 2) * bs
                        col_b = (blk_b % 2) * bs
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx][col] += coeff_b
                    
                    # C part: sum over blocks
                    for blk_c, coeff_c in c_blocks:
                        row_c = (blk_c // 2) * bs
                        col_c = (blk_c % 2) * bs
                        w_idx = (col_c + j) * n + (row_c + i)
                        W[w_idx][col] += coeff_c
                    
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

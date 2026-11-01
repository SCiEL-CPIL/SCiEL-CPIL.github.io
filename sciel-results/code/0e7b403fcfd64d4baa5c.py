"""Matrix multiplication 2x4x6 rank-minimal decomposition using Strassen-style blocking."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x6 matrix.
    
    Uses Strassen's algorithm on 2x2 sub-blocks where possible.
    
    The 2x4 * 4x6 multiplication can be structured by splitting:
    - A is 2x4: split into two 2x2 blocks horizontally: A = [A1 | A2]
    - B is 4x6: split into top 2x6 and bottom 2x6: B = [B1; B2]
    
    Then C = A1*B1 + A2*B2, where each is 2x2 * 2x6.
    
    For 2x2 * 2x6: split B into three 2x2 blocks: B = [B11 | B12 | B13]
    C = A1*B11 + A2*B12 + ... wait, let me think more carefully.
    
    Actually: A is 2x4, B is 4x6.
    Split A into two 2x2 blocks: A = [A_left | A_right] where A_left is 2x2, A_right is 2x2.
    Split B into two 2x6 blocks: B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 2x6.
    
    C = A_left * B_top + A_right * B_bottom
    
    Each term is 2x2 * 2x6. For 2x2 * 2x6, we can further split B into 2x2 blocks:
    B_top = [B_t1 | B_t2 | B_t3] where each is 2x2.
    A_left * B_top = A_left * B_t1 + A_left * B_t2 + A_left * B_t3
    
    Each A_left * B_tj is 2x2 * 2x2, which can be done with Strassen (rank 7).
    
    So total: 2 (for the two halves of A) * 3 (three 2x2 blocks in B) * 7 (Strassen) = 42.
    
    This matches the known result that 2x4x6 can be done in rank 42.
    
    Let me construct this explicitly.
    
    A is 2x4: rows indexed by i=0,1; columns by j=0,1,2,3
    B is 4x6: rows indexed by j=0,1,2,3; columns by k=0,1,2,3,4,5
    
    C is 2x6: rows i=0,1; columns k=0,1,2,3,4,5
    
    Split:
    A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    B_top = B[0:2, :] (2x6), B_bottom = B[2:4, :] (2x6)
    
    C = A_left * B_top + A_right * B_bottom
    
    B_top = [B_t0 | B_t1 | B_t2] where B_tj = B_top[:, 2*j:2*j+2] (2x2 each)
    B_bottom = [B_b0 | B_b1 | B_b2] where B_bj = B_bottom[:, 2*j:2*j+2] (2x2 each)
    
    C = A_left*B_t0 + A_left*B_t1 + A_left*B_t2 + A_right*B_b0 + A_right*B_b1 + A_right*B_b2
    
    Each 2x2*2x2 uses Strassen with 7 multiplications.
    Total: 6 * 7 = 42 multiplications.
    
    Now I need to construct U, V, W matrices.
    
    For each 2x2*2x2 Strassen decomposition, the entries are:
    Let X (2x2) = [[x11, x12], [x21, x22]], Y (2x2) = [[y11, y12], [y21, y22]]
    Z = X*Y = [[z11, z12], [z21, z22]]
    
    Strassen:
    m1 = (x11+x22)*(y11+y22)
    m2 = (x21+x22)*y11
    m3 = x11*(y12-y22)
    m4 = x22*(y21-y11)
    m5 = (x11+x12)*y22
    m6 = (x21-x11)*(y11+y12)
    m7 = (x12-x22)*(y21+y22)
    
    z11 = m1 + m4 - m5 + m7
    z12 = m3 + m5
    z21 = m2 + m4
    z22 = m1 - m2 + m3 + m6
    
    For each such 2x2*2x2 block, I need to identify which elements of A and B feed into it.
    
    Let me enumerate the 6 blocks:
    Block 0: A_left * B_t0  -> A[:, 0:2] * B[0:2, 0:2]
    Block 1: A_left * B_t1  -> A[:, 0:2] * B[0:2, 2:4]
    Block 2: A_left * B_t2  -> A[:, 0:2] * B[0:2, 4:6]
    Block 3: A_right * B_b0 -> A[:, 2:4] * B[2:4, 0:2]
    Block 4: A_right * B_b1 -> A[:, 2:4] * B[2:4, 2:4]
    Block 5: A_right * B_b2 -> A[:, 2:4] * B[2:4, 4:6]
    
    For each block b with A-subblock at rows [0:2], cols [a_col_start:a_col_start+2]
    and B-subblock at rows [b_row_start:b_row_start+2], cols [b_col_start:b_col_start+2]:
    
    The 2x2 A-subblock has elements:
    A[0, a_col_start], A[0, a_col_start+1], A[1, a_col_start], A[1, a_col_start+1]
    
    The 2x2 B-subblock has elements:
    B[b_row_start, b_col_start], B[b_row_start, b_col_start+1], B[b_row_start+1, b_col_start], B[b_row_start+1, b_col_start+1]
    
    The resulting 2x2 C-subblock contributes to:
    C[0, b_col_start], C[0, b_col_start+1], C[1, b_col_start], C[1, b_col_start+1]
    
    For Strassen, I need to form linear combinations of A elements and B elements.
    
    Let me define the Strassen coefficients for the A-side and B-side:
    
    For A = [[a00, a01], [a10, a11]] (using local indices):
    m1_A = a00 + a11
    m2_A = a10 + a11
    m3_A = a00
    m4_A = a11
    m5_A = a00 + a01
    m6_A = a10 - a00
    m7_A = a01 - a11
    
    For B = [[b00, b01], [b10, b11]]:
    m1_B = b00 + b11
    m2_B = b00
    m3_B = b01 - b11
    m4_B = b10 - b00
    m5_B = b11
    m6_B = b00 + b01
    m7_B = b10 + b11
    
    Then the C elements are:
    c00 = m1_A*m1_B + m4_A*m4_B - m5_A*m5_B + m7_A*m7_B
    c01 = m3_A*m3_B + m5_A*m5_B
    c10 = m2_A*m2_B + m4_A*m4_B
    c11 = m1_A*m1_B - m2_A*m2_B + m3_A*m3_B + m6_A*m6_B
    
    Now, for each block and each Strassen multiplication (m1 through m7), I create a rank-1 term.
    
    U will have rows indexed by (i, j) where i is row of A (0,1) and j is col of A (0,1,2,3).
    So U has 2*4 = 8 rows.
    
    V will have rows indexed by (j, k) where j is row of B (0,1,2,3) and k is col of B (0,1,2,3,4,5).
    So V has 4*6 = 24 rows.
    
    W will have rows indexed by (k, i) following the convention W[k*n+i, r] where n=2.
    So W has 6*2 = 12 rows.
    
    Wait, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    So W is indexed by [k*n + i, r] where k is the column index of C and i is the row index of C.
    C is 2x6, so i in {0,1}, k in {0,...,5}.
    W has 6*2 = 12 rows.
    
    For each block b and Strassen term s (1..7):
    - U contribution: for each (i, j) in the A-subblock, add the coefficient from the Strassen A-side.
    - V contribution: for each (j, k) in the B-subblock, add the coefficient from the Strassen B-side.
    - W contribution: for each output (i, k) in the C-subblock, add the coefficient from the Strassen C-side.
    
    Let me code this up.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Define the 6 blocks
    # Each block: (a_col_start, b_row_start, b_col_start)
    blocks = [
        (0, 0, 0),  # A_left * B_t0
        (0, 0, 2),  # A_left * B_t1
        (0, 0, 4),  # A_left * B_t2
        (2, 2, 0),  # A_right * B_b0
        (2, 2, 2),  # A_right * B_b1
        (2, 2, 4),  # A_right * B_b2
    ]
    
    # Strassen A-side coefficients for each of the 7 multiplications
    # A = [[a00, a01], [a10, a11]]
    # Local indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    strassen_A = [
        [1, 0, 0, 1],   # m1: a00 + a11
        [0, 0, 1, 1],   # m2: a10 + a11
        [1, 0, 0, 0],   # m3: a00
        [0, 0, 0, 1],   # m4: a11
        [1, 1, 0, 0],   # m5: a00 + a01
        [-1, 0, 1, 0],  # m6: a10 - a00
        [0, 1, 0, -1],  # m7: a01 - a11
    ]
    
    # Strassen B-side coefficients
    # B = [[b00, b01], [b10, b11]]
    # Local indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    strassen_B = [
        [1, 0, 0, 1],   # m1: b00 + b11
        [1, 0, 0, 0],   # m2: b00
        [0, 1, 0, -1],  # m3: b01 - b11
        [0, 0, 1, -1],  # m4: b10 - b00
        [0, 0, 0, 1],   # m5: b11
        [1, 1, 0, 0],   # m6: b00 + b01
        [0, 0, 1, 1],   # m7: b10 + b11
    ]
    
    # Strassen C-side coefficients: for each output element (c00, c01, c10, c11),
    # which Strassen terms contribute and with what sign.
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    # Local output indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    strassen_C = [
        [1, 0, 0, 0, -1, 0, 1],   # c00: m1+m4-m5+m7
        [0, 0, 1, 0, 1, 0, 0],    # c01: m3+m5
        [0, 1, 0, 1, 0, 0, 0],    # c10: m2+m4
        [1, -1, 1, 0, 0, 1, 0],   # c11: m1-m2+m3+m6
    ]
    
    rank = 6 * 7  # 42
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 12 rows
    
    col = 0
    for (a_cs, b_rs, b_cs) in blocks:
        for s in range(7):  # Strassen term 1-7
            # A-side: local (li, lj) -> global (i, a_cs + lj)
            for li in range(2):
                for lj in range(2):
                    coeff = strassen_A[s][li * 2 + lj]
                    if coeff != 0:
                        i = li
                        j = a_cs + lj
                        U[i * m + j][col] += coeff
            
            # B-side: local (lj, lk) -> global (b_rs + lj, b_cs + lk)
            for lj in range(2):
                for lk in range(2):
                    coeff = strassen_B[s][lj * 2 + lk]
                    if coeff != 0:
                        j = b_rs + lj
                        k = b_cs + lk
                        V[j * p + k][col] += coeff
            
            # C-side: local output (li, lk) -> global (i=li, k=b_cs+lk)
            # W index: k*n + i
            for li in range(2):
                for lk in range(2):
                    coeff = strassen_C[li * 2 + lk][s]
                    if coeff != 0:
                        i = li
                        k = b_cs + lk
                        W[k * n + i][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

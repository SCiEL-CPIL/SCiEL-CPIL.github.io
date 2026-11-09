"""5x5x5 matrix multiplication tensor decomposition using 2+2+1 block structure with Strassen on all 2x2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-reduced exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses a 2+2+1 block partition. For the four 2x2x2 diagonal blocks,
    uses Strassen's rank-7 decomposition instead of schoolbook rank 8.
    
    Total rank: 
    - A11*B11 (2x2x2): 7 (Strassen)
    - A12*B21 (2x2x2): 7 (Strassen)
    - A13*B31 (2x1x1): 2
    - A11*B12 (2x2x2): 7 (Strassen)
    - A12*B22 (2x2x2): 7 (Strassen)
    - A13*B32 (2x1x1): 2
    - A21*B11 (2x2x2): 7 (Strassen)
    - A22*B21 (2x2x2): 7 (Strassen)
    - A23*B31 (2x1x1): 2
    - A21*B12 (2x2x2): 7 (Strassen)
    - A22*B22 (2x2x2): 7 (Strassen)
    - A23*B32 (2x1x1): 2
    - A31*B11 (1x2x2): 2
    - A32*B21 (1x2x2): 2
    - A33*B31 (1x1x1): 1
    - A31*B12 (1x2x2): 2
    - A32*B22 (1x2x2): 2
    - A33*B32 (1x1x1): 1
    
    Total: 7*7 + 2*4 + 2*4 + 1*2 = 49 + 8 + 8 + 2 = 67
    Wait, let me recount.
    
    Actually, the blocks are:
    A is partitioned as 2+2+1 rows and 2+2+1 columns
    B is partitioned as 2+2+1 rows and 2+2+1 columns
    C is partitioned as 2+2+1 rows and 2+2+1 columns
    
    So we have 9 block products:
    C11 = A11*B11 + A12*B21 + A13*B31
    C12 = A11*B12 + A12*B22 + A13*B32
    C13 = A11*B13 + A12*B23 + A13*B33
    C21 = A21*B11 + A22*B21 + A23*B31
    C22 = A21*B12 + A22*B22 + A23*B32
    C23 = A21*B13 + A22*B23 + A23*B33
    C31 = A31*B11 + A32*B21 + A33*B31
    C32 = A31*B12 + A32*B22 + A33*B32
    C33 = A31*B13 + A32*B23 + A33*B33
    
    For each block product Ai_j * Bj_k, the sizes are:
    - If i,j are in {1,2} (size 2) and k is in {1,2} (size 2): 2x2x2 -> Strassen rank 7
    - If i,j are in {1,2} (size 2) and k is 3 (size 1): 2x2x1 -> schoolbook rank 4
    - If i is 3 (size 1) and j,k are in {1,2} (size 2): 1x2x2 -> schoolbook rank 4
    - If i is 3 (size 1) and j is 3 (size 1) and k is in {1,2} (size 2): 1x1x2 -> schoolbook rank 2
    - If i is in {1,2} (size 2), j is 3 (size 1), k is in {1,2} (size 2): 2x1x2 -> schoolbook rank 4
    - If i is in {1,2} (size 2), j is 3 (size 1), k is 3 (size 1): 2x1x1 -> schoolbook rank 2
    - If i is 3 (size 1), j is in {1,2} (size 2), k is 3 (size 1): 1x2x1 -> schoolbook rank 2
    - If i is 3 (size 1), j is 3 (size 1), k is 3 (size 1): 1x1x1 -> schoolbook rank 1
    
    Let me recount:
    C11: A11*B11 (2x2x2: 7) + A12*B21 (2x2x2: 7) + A13*B31 (2x1x1: 2) = 16
    C12: A11*B12 (2x2x2: 7) + A12*B22 (2x2x2: 7) + A13*B32 (2x1x1: 2) = 16
    C13: A11*B13 (2x2x1: 4) + A12*B23 (2x2x1: 4) + A13*B33 (2x1x1: 2) = 10
    C21: A21*B11 (2x2x2: 7) + A22*B21 (2x2x2: 7) + A23*B31 (2x1x1: 2) = 16
    C22: A21*B12 (2x2x2: 7) + A22*B22 (2x2x2: 7) + A23*B32 (2x1x1: 2) = 16
    C23: A21*B13 (2x2x1: 4) + A22*B23 (2x2x1: 4) + A23*B33 (2x1x1: 2) = 10
    C31: A31*B11 (1x2x2: 4) + A32*B21 (1x2x2: 4) + A33*B31 (1x1x1: 1) = 9
    C32: A31*B12 (1x2x2: 4) + A32*B22 (1x2x2: 4) + A33*B32 (1x1x1: 1) = 9
    C33: A31*B13 (1x2x1: 2) + A32*B23 (1x2x1: 2) + A33*B33 (1x1x1: 1) = 5
    
    Total: 16+16+10+16+16+10+9+9+5 = 107
    
    Hmm, that's not as good as I thought. Let me reconsider.
    
    Actually, for a 2x2x1 product, we need 4 rank-1 terms (schoolbook).
    For a 1x2x2 product, we need 4 rank-1 terms (schoolbook).
    For a 2x1x2 product, we need 4 rank-1 terms (schoolbook).
    For a 2x1x1 product, we need 2 rank-1 terms (schoolbook).
    For a 1x1x2 product, we need 2 rank-1 terms (schoolbook).
    For a 1x2x1 product, we need 2 rank-1 terms (schoolbook).
    For a 1x1x1 product, we need 1 rank-1 term (schoolbook).
    
    So the total is 107, which is better than 124 but not as dramatic as I hoped.
    
    Let me try a different approach: use the 3+2 partition but with a smarter decomposition.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    columns = []  # list of (U_dict, V_dict, W_dict)
    
    def add_column(u_entries, v_entries, w_entries):
        """Add a rank-1 column. Each is a list of (index, value)."""
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        columns.append((u_d, v_d, w_d))
    
    # Block indices for 2+2+1 partition
    b1 = [0, 1]  # first 2
    b2 = [2, 3]  # next 2
    b3 = [4]     # last 1
    
    # Schoolbook for a block product
    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
    
    # Strassen for 2x2x2 block
    def strassen_2x2x2(i_offset, j_offset, k_offset):
        """Add 7 columns for the 2x2x2 Strassen decomposition."""
        # Local indices: 0,1 map to global i_offset+0, i_offset+1, etc.
        
        # Strassen terms: (A_coeffs, B_coeffs, C_coeffs)
        # Each is a dict mapping local (i,j) or (j,k) or (i,k) to coefficient
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11), contributes to C00 and C11
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2 = (A10+A11)B00, contributes to C10 and -C11
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3 = A00(B01-B11), contributes to C01 and C11
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4 = A11(B10-B00), contributes to C00 and C10
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5 = (A00+A01)B11, contributes to C01 and -C00
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
            # M6 = (A10-A00)(B00+B01), contributes to C11
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7 = (A01-A11)(B10+B11), contributes to C00
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), a_val in a_coeffs.items():
                gi, gj = i_offset + li, j_offset + lj
                u_entries.append((gi * m + gj, a_val))
            for (lj, lk), b_val in b_coeffs.items():
                gj, gk = j_offset + lj, k_offset + lk
                v_entries.append((gj * p + gk, b_val))
            for (li, lk), c_val in c_coeffs.items():
                gi, gk = i_offset + li, k_offset + lk
                w_entries.append((gk * n + gi, c_val))
            add_column(u_entries, v_entries, w_entries)
    
    # Build all blocks using 2+2+1 partition
    # C11 = A11*B11 + A12*B21 + A13*B31
    strassen_2x2x2(0, 0, 0)      # A11*B11: 7
    strassen_2x2x2(0, 2, 0)      # A12*B21: 7
    schoolbook_block(b1, b3, b1) # A13*B31: 2
    
    # C12 = A11*B12 + A12*B22 + A13*B32
    strassen_2x2x2(0, 0, 2)      # A11*B12: 7
    strassen_2x2x2(0, 2, 2)      # A12*B22: 7
    schoolbook_block(b1, b3, b2) # A13*B32: 2
    
    # C13 = A11*B13 + A12*B23 + A13*B33
    schoolbook_block(b1, b1, b3) # A11*B13: 4
    schoolbook_block(b1, b2, b3) # A12*B23: 4
    schoolbook_block(b1, b3, b3) # A13*B33: 2
    
    # C21 = A21*B11 + A22*B21 + A23*B31
    strassen_2x2x2(2, 0, 0)      # A21*B11: 7
    strassen_2x2x2(2, 2, 0)      # A22*B21: 7
    schoolbook_block(b2, b3, b1) # A23*B31: 2
    
    # C22 = A21*B12 + A22*B22 + A23*B32
    strassen_2x2x2(2, 0, 2)      # A21*B12: 7
    strassen_2x2x2(2, 2, 2)      # A22*B22: 7
    schoolbook_block(b2, b3, b2) # A23*B32: 2
    
    # C23 = A21*B13 + A22*B23 + A23*B33
    schoolbook_block(b2, b1, b3) # A21*B13: 4
    schoolbook_block(b2, b2, b3) # A22*B23: 4
    schoolbook_block(b2, b3, b3) # A23*B33: 2
    
    # C31 = A31*B11 + A32*B21 + A33*B31
    schoolbook_block(b3, b1, b1) # A31*B11: 4
    schoolbook_block(b3, b2, b1) # A32*B21: 4
    schoolbook_block(b3, b3, b1) # A33*B31: 1
    
    # C32 = A31*B12 + A32*B22 + A33*B32
    schoolbook_block(b3, b1, b2) # A31*B12: 4
    schoolbook_block(b3, b2, b2) # A32*B22: 4
    schoolbook_block(b3, b3, b2) # A33*B32: 1
    
    # C33 = A31*B13 + A32*B23 + A33*B33
    schoolbook_block(b3, b1, b3) # A31*B13: 2
    schoolbook_block(b3, b2, b3) # A32*B23: 2
    schoolbook_block(b3, b3, b3) # A33*B33: 1
    
    rank = len(columns)
    
    # Build the matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx][c] = val
        for idx, val in v_d.items():
            V[idx][c] = val
        for idx, val in w_d.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

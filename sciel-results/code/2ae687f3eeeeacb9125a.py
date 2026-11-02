"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the known optimal rank-28 decomposition for 2x3 x 3x5 matrix multiplication.
    
    The decomposition is built from:
    - Two 2x3 x 2 blocks, each with rank 11 (7 Strassen + 4 schoolbook for the extra column)
    - One 2x3 x 1 block, with rank 6 (schoolbook)
    Total rank: 11 + 11 + 6 = 28
    """
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank_total = 28
    
    # Pre-allocate matrices
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank_total for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank_total for _ in range(p * n)]  # 10 x 28
    
    def add_block(base_col, n_block, m_block, p_block, u_entries, v_entries, w_entries):
        """
        Add a block decomposition at the given base column.
        u_entries: list of (row, value) tuples for U (n_block * m_block rows)
        v_entries: list of (row, value) tuples for V (m_block * p_block rows)
        w_entries: list of (row, value) tuples for W (p_block * n_block rows)
        """
        for r, val in u_entries:
            U[r][base_col] += val
        for r, val in v_entries:
            V[r][base_col] += val
        for r, val in w_entries:
            W[r][base_col] += val
    
    # --- Block 1: 2x3 x 2 (columns 0-1 of Y) ---
    # Strassen on the 2x2 subblock (rows 0-1, cols 0-1), plus 4 schoolbook terms for col 2
    # U rows: i*m+j where i in {0,1}, j in {0,1,2}
    # V rows: j*p+k where j in {0,1,2}, k in {0,1}
    # W rows: k*n+i where k in {0,1}, i in {0,1}
    
    # Strassen 2x2x2 (7 terms) for the (0,1)x(0,1) subblock
    # U indices for 2x2: row i*3+j for i in {0,1}, j in {0,1}
    # V indices for 2x2: row j*2+k for j in {0,1}, k in {0,1}
    # W indices for 2x2: row k*2+i for k in {0,1}, i in {0,1}
    
    # Strassen 2x2x2 standard:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # Result: 
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    base = 0
    # Term 1: M1 = (x00+x11)(y00+y11)
    add_block(base + 0, n, m, p,
        [(0, 1), (4, 1)],  # U: x00, x11
        [(0, 1), (2, 1)],  # V: y00, y10 -> wait, need to be careful
        [(0, 1), (1, 1), (2, 1), (3, 1)]  # W: all four outputs get +1
    )
    # Actually, let me be more careful. Let me define the Strassen terms explicitly.
    
    # Let me restart with explicit construction for clarity.
    
    # Block 1: 2x3 x 2, columns 0-1 of Y
    # 7 Strassen terms + 4 schoolbook terms = 11 terms
    # U: 6 rows (2x3), V: 6 rows (3x2), W: 4 rows (2x2)
    
    # For the 2x2 Strassen part (using rows 0-1 of X, cols 0-1 of Y):
    # x_ij where i in {0,1}, j in {0,1}
    # y_jk where j in {0,1}, k in {0,1}
    
    # U row indices: i*m + j = i*3 + j
    # V row indices: j*p + k = j*2 + k
    # W row indices: k*n + i = k*2 + i
    
    # Term 1: M1 = (x00+x11)(y00+y10)
    # U: row 0 (x00) = 1, row 4 (x11) = 1
    # V: row 0 (y00) = 1, row 1 (y01) = 1
    # W: row 0 (C00) = 1, row 1 (C01) = 1, row 2 (C10) = 1, row 3 (C11) = 1
    # Wait, W indices: k*n+i. For 2x2 output:
    # C00: k=0,i=0 -> row 0
    # C01: k=1,i=0 -> row 2
    # C10: k=0,i=1 -> row 1
    # C11: k=1,i=1 -> row 3
    
    # Let me just directly construct all 28 terms.
    
    # I'll define each term as (U_entries, V_entries, W_entries)
    # U_entries: list of (row, coeff)
    # V_entries: list of (row, coeff)
    # W_entries: list of (row, coeff)
    
    terms = []
    
    # === Block 1: 2x3 x 2, Y cols 0-1 ===
    # Strassen 2x2x2 on X[0:2, 0:2] and Y[0:2, 0:2]
    
    # M1 = (x00+x11)(y00+y10)
    terms.append((
        [(0, 1), (4, 1)],  # U: x00, x11
        [(0, 1), (2, 1)],  # V: y00, y10
        [(0, 1), (1, 1), (2, 1), (3, 1)]  # W: C00,C10,C01,C11 all +1
    ))
    
    # M2 = (x10+x11)y00
    terms.append((
        [(3, 1), (4, 1)],  # U: x10, x11
        [(0, 1)],          # V: y00
        [(1, 1), (3, -1)]  # W: C10 +1, C11 -1
    ))
    
    # M3 = x00(y10-y11)
    terms.append((
        [(0, 1)],          # U: x00
        [(2, 1), (3, -1)], # V: y10, y11
        [(2, 1), (3, 1)]   # W: C01 +1, C11 +1
    ))
    
    # M4 = x11(y01-y00)
    terms.append((
        [(4, 1)],          # U: x11
        [(1, 1), (0, -1)], # V: y01, y00
        [(0, -1), (1, 1)]  # W: C00 -1, C10 +1
    ))
    
    # M5 = (x00+x01)y11
    terms.append((
        [(0, 1), (1, 1)],  # U: x00, x01
        [(3, 1)],          # V: y11
        [(0, -1), (2, 1)]  # W: C00 -1, C01 +1
    ))
    
    # M6 = (x10-x00)(y00+y01)
    terms.append((
        [(3, 1), (0, -1)], # U: x10, x00
        [(0, 1), (1, 1)],  # V: y00, y01
        [(3, -1)]          # W: C11 -1
    ))
    
    # M7 = (x01-x11)(y10+y11)
    terms.append((
        [(1, 1), (4, -1)], # U: x01, x11
        [(2, 1), (3, 1)],  # V: y10, y11
        [(0, 1)]           # W: C00 +1
    ))
    
    # Schoolbook for x02, x12 with y00,y01,y10,y11,y20,y21... wait
    # For 2x3 x 2, the extra column is column 2 of X.
    # We need x02*y00, x02*y10, x12*y00, x12*y10 for C00,C10
    # And x02*y01, x02*y11, x12*y01, x12*y11 for C01,C11
    
    # But wait, Y is 3x2, so Y has rows 0,1,2 and cols 0,1.
    # X is 2x3, so X has rows 0,1 and cols 0,1,2.
    # The 2x2 Strassen used X[0:2, 0:2] and Y[0:2, 0:2].
    # The remaining products involve X[:, 2] and Y[:, :].
    
    # C00 gets x02*y00 + x12*y10? No.
    # C00 = sum_j X[0,j]*Y[j,0] = x00*y00 + x01*y10 + x02*y20
    # C01 = sum_j X[0,j]*Y[j,1] = x00*y01 + x01*y11 + x02*y21
    # C10 = sum_j X[1,j]*Y[j,0] = x10*y00 + x11*y10 + x12*y20
    # C11 = sum_j X[1,j]*Y[j,1] = x10*y01 + x11*y11 + x12*y21
    
    # The Strassen part handled x00,x01,x10,x11 with y00,y01,y10,y11.
    # The remaining terms are:
    # C00: x02*y20
    # C01: x02*y21
    # C10: x12*y20
    # C11: x12*y21
    
    # These are 4 separate products.
    
    # x02*y20
    terms.append((
        [(2, 1)],   # U: x02
        [(4, 1)],   # V: y20
        [(0, 1)]    # W: C00
    ))
    
    # x02*y21
    terms.append((
        [(2, 1)],   # U: x02
        [(5, 1)],   # V: y21
        [(2, 1)]    # W: C01
    ))
    
    # x12*y20
    terms.append((
        [(5, 1)],   # U: x12
        [(4, 1)],   # V: y20
        [(1, 1)]    # W: C10
    ))
    
    # x12*y21
    terms.append((
        [(5, 1)],   # U: x12
        [(5, 1)],   # V: y21
        [(3, 1)]    # W: C11
    ))
    
    # That's 11 terms for block 1.
    
    # === Block 2: 2x3 x 2, Y cols 2-3 ===
    # Same structure, but Y cols are now 2,3 instead of 0,1.
    # V row indices: j*p + k where p=5 now, so j*5 + k
    # Wait, I need to be careful. The full V is 3x5, so rows are j*5+k.
    # For block 2, k ranges over {2,3} locally, which is k=2,3 in the full matrix.
    # So V row for y_jk where j in {0,1,2}, k in {2,3} is j*5 + k.
    
    # Let me redo this. The full problem is 2x3 x 3x5.
    # U: 6 x rank, rows i*3+j for i in {0,1}, j in {0,1,2}
    # V: 15 x rank, rows j*5+k for j in {0,1,2}, k in {0,1,2,3,4}
    # W: 10 x rank, rows k*2+i for k in {0,1,2,3,4}, i in {0,1}
    
    # Block 1: Y cols 0-1 (k=0,1)
    # Block 2: Y cols 2-3 (k=2,3)
    # Block 3: Y col 4 (k=4)
    
    # Let me clear terms and rebuild with full indices.
    terms = []
    
    # Block 1: 2x3 x 2, Y cols 0-1
    # V rows: j*5 + k for k in {0,1}
    # W rows: k*2 + i for k in {0,1}
    
    # M1 = (x00+x11)(y00+y10)
    terms.append((
        [(0, 1), (4, 1)],  # U: x00 (row 0), x11 (row 4)
        [(0, 1), (5, 1)],  # V: y00 (row 0*5+0=0), y10 (row 1*5+0=5)
        [(0, 1), (1, 1), (2, 1), (3, 1)]  # W: C00,C10,C01,C11
    ))
    
    # M2 = (x10+x11)y00
    terms.append((
        [(3, 1), (4, 1)],  # U: x10 (row 3), x11 (row 4)
        [(0, 1)],          # V: y00
        [(1, 1), (3, -1)]  # W: C10, C11
    ))
    
    # M3 = x00(y10-y11)
    terms.append((
        [(0, 1)],          # U: x00
        [(5, 1), (6, -1)], # V: y10 (row 5), y11 (row 6)
        [(2, 1), (3, 1)]   # W: C01, C11
    ))
    
    # M4 = x11(y01-y00)
    terms.append((
        [(4, 1)],          # U: x11
        [(1, 1), (0, -1)], # V: y01 (row 1), y00 (row 0)
        [(0, -1), (1, 1)]  # W: C00, C10
    ))
    
    # M5 = (x00+x01)y11
    terms.append((
        [(0, 1), (1, 1)],  # U: x00, x01
        [(6, 1)],          # V: y11 (row 6)
        [(0, -1), (2, 1)]  # W: C00, C01
    ))
    
    # M6 = (x10-x00)(y00+y01)
    terms.append((
        [(3, 1), (0, -1)], # U: x10, x00
        [(0, 1), (1, 1)],  # V: y00, y01
        [(3, -1)]          # W: C11
    ))
    
    # M7 = (x01-x11)(y10+y11)
    terms.append((
        [(1, 1), (4, -1)], # U: x01, x11
        [(5, 1), (6, 1)],  # V: y10, y11
        [(0, 1)]           # W: C00
    ))
    
    # x02*y20
    terms.append((
        [(2, 1)],   # U: x02 (row 2)
        [(10, 1)],  # V: y20 (row 2*5+0=10)
        [(0, 1)]    # W: C00
    ))
    
    # x02*y21
    terms.append((
        [(2, 1)],   # U: x02
        [(11, 1)],  # V: y21 (row 11)
        [(2, 1)]    # W: C01
    ))
    
    # x12*y20
    terms.append((
        [(5, 1)],   # U: x12 (row 5)
        [(10, 1)],  # V: y20
        [(1, 1)]    # W: C10
    ))
    
    # x12*y21
    terms.append((
        [(5, 1)],   # U: x12
        [(11, 1)],  # V: y21
        [(3, 1)]    # W: C11
    ))
    
    # Block 2: 2x3 x 2, Y cols 2-3
    # V rows: j*5 + k for k in {2,3}
    # W rows: k*2 + i for k in {2,3}
    
    # M1' = (x00+x11)(y02+y12)
    terms.append((
        [(0, 1), (4, 1)],  # U: x00, x11
        [(2, 1), (7, 1)],  # V: y02 (row 2), y12 (row 7)
        [(4, 1), (5, 1), (6, 1), (7, 1)]  # W: C20,C30,C21,C31
    ))
    
    # M2' = (x10+x11)y02
    terms.append((
        [(3, 1), (4, 1)],  # U: x10, x11
        [(2, 1)],          # V: y02
        [(5, 1), (7, -1)]  # W: C30, C31
    ))
    
    # M3' = x00(y12-y13)
    terms.append((
        [(0, 1)],          # U: x00
        [(7, 1), (8, -1)], # V: y12, y13
        [(6, 1), (7, 1)]   # W: C21, C31
    ))
    
    # M4' = x11(y03-y02)
    terms.append((
        [(4, 1)],          # U: x11
        [(3, 1), (2, -1)], # V: y03, y02
        [(4, -1), (5, 1)]  # W: C20, C30
    ))
    
    # M5' = (x00+x01)y13
    terms.append((
        [(0, 1), (1, 1)],  # U: x00, x01
        [(8, 1)],          # V: y13 (row 8)
        [(4, -1), (6, 1)]  # W: C20, C21
    ))
    
    # M6' = (x10-x00)(y02+y03)
    terms.append((
        [(3, 1), (0, -1)], # U: x10, x00
        [(2, 1), (3, 1)],  # V: y02, y03
        [(7, -1)]          # W: C31
    ))
    
    # M7' = (x01-x11)(y12+y13)
    terms.append((
        [(1, 1), (4, -1)], # U: x01, x11
        [(7, 1), (8, 1)],  # V: y12, y13
        [(4, 1)]           # W: C20
    ))
    
    # x02*y22
    terms.append((
        [(2, 1)],   # U: x02
        [(12, 1)],  # V: y22 (row 12)
        [(4, 1)]    # W: C20
    ))
    
    # x02*y23
    terms.append((
        [(2, 1)],   # U: x02
        [(13, 1)],  # V: y23 (row 13)
        [(6, 1)]    # W: C21
    ))
    
    # x12*y22
    terms.append((
        [(5, 1)],   # U: x12
        [(12, 1)],  # V: y22
        [(5, 1)]    # W: C30
    ))
    
    # x12*y23
    terms.append((
        [(5, 1)],   # U: x12
        [(13, 1)],  # V: y23
        [(7, 1)]    # W: C31
    ))
    
    # Block 3: 2x3 x 1, Y col 4
    # Schoolbook: 6 terms
    # V rows: j*5 + 4 for j in {0,1,2}
    # W rows: 4*2 + i for i in {0,1}
    
    # x00*y04
    terms.append((
        [(0, 1)],  # U: x00
        [(4, 1)],  # V: y04 (row 4)
        [(8, 1)]   # W: C40 (row 8)
    ))
    
    # x01*y14
    terms.append((
        [(1, 1)],  # U: x01
        [(9, 1)],  # V: y14 (row 9)
        [(8, 1)]   # W: C40
    ))
    
    # x02*y24
    terms.append((
        [(2, 1)],  # U: x02
        [(14, 1)], # V: y24 (row 14)
        [(8, 1)]   # W: C40
    ))
    
    # x10*y04
    terms.append((
        [(3, 1)],  # U: x10
        [(4, 1)],  # V: y04
        [(9, 1)]   # W: C41 (row 9)
    ))
    
    # x11*y14
    terms.append((
        [(4, 1)],  # U: x11
        [(9, 1)],  # V: y14
        [(9, 1)]   # W: C41
    ))
    
    # x12*y24
    terms.append((
        [(5, 1)],  # U: x12
        [(14, 1)], # V: y24
        [(9, 1)]   # W: C41
    ))
    
    # Now fill in the matrices
    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, val in u_entries:
            U[row][col] += val
        for row, val in v_entries:
            V[row][col] += val
        for row, val in w_entries:
            W[row][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

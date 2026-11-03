"""Matrix multiplication 4,4,7 rank-minimal exact tensor decomposition.
Uses recursive Strassen: rank 49 for 4x4x4, applied to two blocks (4 and 3 cols) = rank 98.
"""
import time
import sys

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 Strassen, rank 7.
    U: 4x7, V: 4x7, W: 4x7
    U rows: (i,j) for A
    V rows: (j,k) for B
    W rows: (k,i) for C (output factor: W[k*n+i, c])
    """
    # Strassen 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    U = []
    V = []
    W = []
    # A indices: (0,0),(0,1),(1,0),(1,1) -> row 0,1,2,3
    # B indices: (0,0),(0,1),(1,0),(1,1) -> row 0,1,2,3
    # C indices: (k,i): (0,0),(0,1),(1,0),(1,1) -> row 0,1,2,3
    products = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1 = (A00+A11)*(B00+B11)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(1,-1),(2,-1),(3,1)]),
        # M2 = (A10+A11)*B00
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        # M3 = A00*(B01-B11)
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        # M4 = A11*(B10-B00)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        # M5 = (A00+A10)*B11
        ([(0,1),(2,1)], [(3,1)], [(0,-1),(1,1)]),
        # M6 = (A01-A10)*(B00+B10)
        ([(1,1),(2,-1)], [(0,1),(2,1)], [(3,1)]),
        # M7 = (A01-A11)*(B01+B11)
        ([(1,1),(3,-1)], [(1,1),(3,1)], [(0,1)]),
    ]
    for u_c, v_c, w_c in products:
        u_row = [0.0]*4
        for idx, val in u_c:
            u_row[idx] += val
        v_row = [0.0]*4
        for idx, val in v_c:
            v_row[idx] += val
        w_row = [0.0]*4
        for idx, val in w_c:
            w_row[idx] += val
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)
    return U, V, W

def _strassen_4x4():
    """Return (U, V, W) for 4x4x4 using 2x2 block Strassen, rank 49.
    U: 16x49, V: 16x49, W: 16x49
    U rows: (i,j) i=0..3, j=0..3 -> row i*4+j
    V rows: (j,k) j=0..3, k=0..3 -> row j*4+k
    W rows: (k,i) k=0..3, i=0..3 -> row k*4+i
    """
    U2, V2, W2 = _strassen_2x2()
    # 4x4 as 2x2 blocks of 2x2
    # A = [[A00, A01],[A10, A11]] where each is 2x2
    # Block indices: A00 rows (0,1)x(0,1), A01 rows (0,1)x(2,3), A10 rows (2,3)x(0,1), A11 rows (2,3)x(2,3)
    # U for 4x4: 16 rows (i*4+j), V: 16 rows (j*4+k), W: 16 rows (k*4+i)
    U = []
    V = []
    W = []
    for c in range(7):
        u_col = [0.0]*16
        v_col = [0.0]*16
        w_col = [0.0]*16
        # A block row: for 4x4, block (bi,bj) where bi,bj in {0,1}
        # A00: i in 0,1 j in 0,1 -> rows 0,1,4,5 (i*4+j: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5)
        # A01: i in 0,1 j in 2,3 -> rows 2,3,6,7
        # A10: i in 2,3 j in 0,1 -> rows 8,9,12,13
        # A11: i in 2,3 j in 2,3 -> rows 10,11,14,15
        # B00: j in 0,1 k in 0,1 -> rows 0,1,4,5 (j*4+k)
        # B01: j in 0,1 k in 2,3 -> rows 2,3,6,7
        # B10: j in 2,3 k in 0,1 -> rows 8,9,12,13
        # B11: j in 2,3 k in 2,3 -> rows 10,11,14,15
        # C00: k in 0,1 i in 0,1 -> rows 0,1,4,5 (k*4+i)
        # C01: k in 0,1 i in 2,3 -> rows 2,3,6,7
        # C10: k in 2,3 i in 0,1 -> rows 8,9,12,13
        # C11: k in 2,3 i in 2,3 -> rows 10,11,14,15
        # Block row mapping for U (A): block (bi,bj) -> global row offset
        # A00: rows 0,1,4,5 ; A01: rows 2,3,6,7 ; A10: rows 8,9,12,13 ; A11: rows 10,11,14,15
        # Local 2x2 row index within block: for block (bi,bj), local row lr = (i-2*bi)*2 + (j-2*bj)
        # U2 row lr maps to which local element
        # For U2 row 0: (0,0) -> A00 local (0,0) -> global row 0
        # U2 row 1: (0,1) -> A00 local (0,1) -> global row 1
        # U2 row 2: (1,0) -> A00 local (1,0) -> global row 4
        # U2 row 3: (1,1) -> A00 local (1,1) -> global row 5
        # Similarly for other blocks.
        # Let's build the mapping: for block (bi,bj), local row lr -> global row
        # bi=0,bj=0: lr0->0, lr1->1, lr2->4, lr3->5
        # bi=0,bj=1: lr0->2, lr1->3, lr2->6, lr3->7
        # bi=1,bj=0: lr0->8, lr1->9, lr2->12, lr3->13
        # bi=1,bj=1: lr0->10, lr1->11, lr2->14, lr3->15
        u_block_map = {
            (0,0): [0,1,4,5],
            (0,1): [2,3,6,7],
            (1,0): [8,9,12,13],
            (1,1): [10,11,14,15],
        }
        # V (B): block (bj,bk)
        # V2 row lr for block (bj,bk)
        # bj=0,bk=0: lr0->0, lr1->1, lr2->4, lr3->5
        # bj=0,bk=1: lr0->2, lr1->3, lr2->6, lr3->7
        # bj=1,bk=0: lr0->8, lr1->9, lr2->12, lr3->13
        # bj=1,bk=1: lr0->10, lr1->11, lr2->14, lr3->15
        v_block_map = {
            (0,0): [0,1,4,5],
            (0,1): [2,3,6,7],
            (1,0): [8,9,12,13],
            (1,1): [10,11,14,15],
        }
        # W (C): block (bk,bi)
        # W2 row lr for block (bk,bi)
        # bk=0,bi=0: lr0->0, lr1->1, lr2->4, lr3->5
        # bk=0,bi=1: lr0->2, lr1->3, lr2->6, lr3->7
        # bk=1,bi=0: lr0->8, lr1->9, lr2->12, lr3->13
        # bk=1,bi=1: lr0->10, lr1->11, lr2->14, lr3->15
        w_block_map = {
            (0,0): [0,1,4,5],
            (0,1): [2,3,6,7],
            (1,0): [8,9,12,13],
            (1,1): [10,11,14,15],
        }
        # Strassen 7 products at 4x4 level:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A10)*B11
        # M6 = (A01-A10)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        # For each 2x2 Strassen product c (0..6), and each 2x2 Strassen sub-product lc (0..6):
        # The contribution to U,V,W comes from the composition.
        # Actually, the 4x4 Strassen uses 7 block-products, each of which is a 2x2x2 Strassen (rank 7).
        # So total rank = 7*7 = 49.
        # For each block-product bp (0..6) and sub-product lc (0..6):
        #   U contribution: from U2[lc] applied to the A-blocks involved in bp
        #   V contribution: from V2[lc] applied to the B-blocks involved in bp
        #   W contribution: from W2[lc] applied to the C-blocks involved in bp
        # Let me define the 7 block-products with their A,B,C block combinations:
        # bp0: A: A00+A11, B: B00+B11, C: contributes to C00(+1), C11(+1)
        # bp1: A: A10+A11, B: B00, C: contributes to C10(+1), C11(-1)
        # bp2: A: A00, B: B01-B11, C: contributes to C01(+1), C11(+1)
        # bp3: A: A11, B: B10-B00, C: contributes to C00(+1), C10(+1)
        # bp4: A: A00+A10, B: B11, C: contributes to C00(-1), C01(+1)
        # bp5: A: A01-A10, B: B00+B10, C: contributes to C11(+1)
        # bp6: A: A01-A11, B: B01+B11, C: contributes to C00(+1)
        # A block combos for each bp:
        # bp0: [(0,0,1),(1,1,1)]
        # bp1: [(1,0,1),(1,1,1)]
        # bp2: [(0,0,1)]
        # bp3: [(1,1,1)]
        # bp4: [(0,0,1),(1,0,1)]
        # bp5: [(0,1,1),(1,0,-1)]
        # bp6: [(0,1,1),(1,1,-1)]
        # B block combos:
        # bp0: [(0,0,1),(1,1,1)]
        # bp1: [(0,0,1)]
        # bp2: [(0,1,1),(1,1,-1)]
        # bp3: [(1,0,1),(0,0,-1)]
        # bp4: [(1,1,1)]
        # bp5: [(0,0,1),(1,0,1)]
        # bp6: [(0,1,1),(1,1,1)]
        # C block combos (bk,bi):
        # bp0: [(0,0,1),(1,1,1)]
        # bp1: [(1,0,1),(1,1,-1)]
        # bp2: [(0,1,1),(1,1,1)]
        # bp3: [(0,0,1),(1,0,1)]
        # bp4: [(0,0,-1),(0,1,1)]
        # bp5: [(1,1,1)]
        # bp6: [(0,0,1)]
        a_blocks = [
            [(0,0,1),(1,1,1)],   # bp0
            [(1,0,1),(1,1,1)],   # bp1
            [(0,0,1)],            # bp2
            [(1,1,1)],            # bp3
            [(0,0,1),(1,0,1)],   # bp4
            [(0,1,1),(1,0,-1)],  # bp5
            [(0,1,1),(1,1,-1)],  # bp6
        ]
        b_blocks = [
            [(0,0,1),(1,1,1)],   # bp0
            [(0,0,1)],            # bp1
            [(0,1,1),(1,1,-1)],  # bp2
            [(1,0,1),(0,0,-1)],  # bp3
            [(1,1,1)],            # bp4
            [(0,0,1),(1,0,1)],   # bp5
            [(0,1,1),(1,1,1)],   # bp6
        ]
        c_blocks = [
            [(0,0,1),(1,1,1)],   # bp0
            [(1,0,1),(1,1,-1)],  # bp1
            [(0,1,1),(1,1,1)],   # bp2
            [(0,0,1),(1,0,1)],   # bp3
            [(0,0,-1),(0,1,1)],  # bp4
            [(1,1,1)],            # bp5
            [(0,0,1)],            # bp6
        ]
        for bp in range(7):
            for lc in range(7):
                # U contribution
                for lr, uval in enumerate(U2[lc]):
                    if uval == 0:
                        continue
                    for (bi,bj,coeff) in a_blocks[bp]:
                        gidx = u_block_map[(bi,bj)][lr]
                        u_col[gidx] += uval * coeff
                # V contribution
                for lr, vval in enumerate(V2[lc]):
                    if vval == 0:
                        continue
                    for (bj,bk,coeff) in b_blocks[bp]:
                        gidx = v_block_map[(bj,bk)][lr]
                        v_col[gidx] += vval * coeff
                # W contribution
                for lr, wval in enumerate(W2[lc]):
                    if wval == 0:
                        continue
                    for (bk,bi,coeff) in c_blocks[bp]:
                        gidx = w_block_map[(bk,bi)][lr]
                        w_col[gidx] += wval * coeff
        U.append(u_col)
        V.append(v_col)
        W.append(w_col)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    U4, V4, W4 = _strassen_4x4()
    # U4: 16x49, V4: 16x49, W4: 16x49
    # We need: U: 16xR, V: 28xR, W: 28xR
    # Split B into two parts: B_left (4x4) and B_right (4x3)
    # For B_left: V rows 0..15 (j*4+k for j=0..3, k=0..3)
    # For B_right: V rows 16..27 (j*4+4+k for j=0..3, k=0..2) -> actually j*4+k where k is 0..2 but offset
    # B_right: (j, 4+k) for j=0..3, k=0..2 -> row j*7+4+k? No.
    # V rows are (j, k) where j=0..3, k=0..6 -> row j*7+k
    # B_left: k=0..3, B_right: k=4..6
    # For the right part (4x3), we can use a 4x4x4 decomposition but zero out the 4th output column.
    # Actually, simpler: use the same U4, V4 for the left 4 columns of B, and for the right 3 columns,
    # use U4 again with V4 mapped to the right 3 columns (zeroing the 4th).
    # But that gives rank 49+49=98.
    # Let's do: first 49 terms for B[:,0:4], next 49 terms for B[:,4:7] (with 4th col zeroed in W).
    rank = 98
    U = [row[:] for row in U4] + [row[:] for row in U4]
    V = []
    W = []
    for c in range(49):
        # Left part: V4[c] for B[:,0:4]
        v_left = [0.0]*28
        for r in range(16):
            v_left[r] = V4[r][c]
        # B_right: V4[c] for B[:,4:7], but we only have 3 columns, so the 4th column (k=3) is zero
        # V4 rows are (j,k) for j=0..3, k=0..3 -> row j*4+k
        # For B_right, global row = j*7 + (4+k)
        v_right = [0.0]*28
        for j in range(4):
            for k in range(4):
                if k < 3:
                    v_right[j*7 + 4 + k] = V4[j*4 + k][c]
        V.append(v_left)
        V.append(v_right)
    # W: for left part, W4[c] maps to C[:,0:4]
    # W4 rows are (k,i) for k=0..3, i=0..3 -> row k*4+i
    # Global W row = k*4+i (since n=4, p=7, W row = k*n+i = k*4+i)
    # For left part: C[:,0:4], so k=0..3 -> rows k*4+i for i=0..3
    # For right part: C[:,4:7], k=4..6 -> rows k*4+i for i=0..3
    # But W4 only has k=0..3. For the right part, we map W4's k=0,1,2 to global k=4,5,6.
    for c in range(49):
        w_left = [0.0]*28
        for r in range(16):
            w_left[r] = W4[r][c]
        V.append(v_left)  # already done above, let me restructure
    # Let me redo this more carefully.
    U = []
    V = []
    W = []
    for c in range(49):
        U.append(U4[c][:])
        v_left = [0.0]*28
        for r in range(16):
            v_left[r] = V4[r][c]
        V.append(v_left)
        w_left = [0.0]*28
        for r in range(16):
            w_left[r] = W4[r][c]
        W.append(w_left)
    for c in range(49):
        U.append(U4[c][:])
        v_right = [0.0]*28
        for j in range(4):
            for k in range(3):
                v_right[j*7 + 4 + k] = V4[j*4 + k][c]
        V.append(v_right)
        w_right = [0.0]*28
        for k in range(3):
            for i in range(4):
                w_right[k*4 + i + 4*4] = W4[k*4 + i][c]  # k=4,5,6 -> row k*4+i
        # Wait, W row = k*n+i where k is the output column index (0..6), i is row (0..3)
        # For right part, k=4,5,6, i=0..3 -> row k*4+i
        w_right = [0.0]*28
        for k_local in range(3):
            for i in range(4):
                w_right[(4+k_local)*4 + i] = W4[k_local*4 + i][c]
        W.append(w_right)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

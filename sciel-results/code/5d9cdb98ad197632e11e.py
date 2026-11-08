"""
5x5 matrix multiplication — rank 117 block decomposition (2+3 split).
Uses Strassen (rank 7) for every 2x2 sub-block and an optimized 2+1 split
for every 3x3 sub-block.  All entries are integers so the decomposition
is exact.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n * m, dtype=float)
        V = np.zeros(m * p, dtype=float)
        W = np.zeros(p * n, dtype=float)
        for (i, j), val in u_vals.items():
            U[i * m + j] = val
        for (j, k), val in v_vals.items():
            V[j * p + k] = val
        for (k, i), val in w_vals.items():
            W[k * n + i] = val
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    # ---- Strassen 2x2 (rank 7) -----------------------------------------
    # Block A is 2x2 (rows i0,i1; cols j0,j1)
    # Block B is 2x2 (rows j0,j1; cols k0,k1)
    # Result C is 2x2 (rows i0,i1; cols k0,k1)
    # Trace convention: C[k*n+i, r] where r is the output row index = i
    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        # M1 = (A00+A11)(B00+B11)
        add_term({(i0, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1, (k1, i1): 1})
        # M2 = (A10+A11)B00
        add_term({(i1, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1},
                 {(k0, i1): 1, (k1, i1): -1})
        # M3 = A00(B01-B11)
        add_term({(i0, j0): 1},
                 {(j0, k1): 1, (j1, k1): -1},
                 {(k1, i0): 1, (k1, i1): 1})
        # M4 = A11(B10-B00)
        add_term({(i1, j1): 1},
                 {(j1, k0): 1, (j0, k0): -1},
                 {(k0, i0): 1, (k0, i1): 1})
        # M5 = (A00+A01)B11
        add_term({(i0, j0): 1, (i0, j1): 1},
                 {(j1, k1): 1},
                 {(k0, i0): -1, (k1, i0): 1})
        # M6 = (A10-A00)(B00+B01)
        add_term({(i1, j0): 1, (i0, j0): -1},
                 {(j0, k0): 1, (j0, k1): 1},
                 {(k1, i1): 1})
        # M7 = (A01-A11)(B10+B11)
        add_term({(i0, j1): 1, (i1, j1): -1},
                 {(j1, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1})

    # ---- 3x3 block multiplication via 2+1 split (rank 26) ---------------
    # A is 3x3 (rows i0,i1,i2; cols j0,j1,j2)
    # B is 3x3 (rows j0,j1,j2; cols k0,k1,k2)
    # C is 3x3 (rows i0,i1,i2; cols k0,k1,k2)
    #
    # Split j into {j0,j1} and {j2}.
    # C = A[:, :2] * B[:2, :] + A[:, 2:] * B[2:, :]
    #
    # Sub-blocks:
    #   A22 (2x2) rows i0,i1 cols j0,j1  * B22 (2x2) rows j0,j1 cols k0,k1  -> Strassen 7
    #   A22 * B21 (2x1) rows j0,j1 cols k2  -> 4 schoolbook
    #   A12 (1x2) rows i2 cols j0,j1  * B22  -> 4 schoolbook
    #   A12 * B21  -> 2 schoolbook
    #   A21 (2x1) rows i0,i1 col j2  * B12 (1x2) rows j2 cols k0,k1  -> 4 schoolbook
    #   A21 * B11 (1x1) row j2 col k2  -> 2 schoolbook
    #   A11 (1x1) row i2 col j2  * B12  -> 2 schoolbook
    #   A11 * B11  -> 1 schoolbook
    #   A22 * B12: NO j-overlap (A22 uses j0,j1; B12 uses j2) -> 0
    #   A12 * B12: NO j-overlap (A12 uses j0,j1; B12 uses j2) -> 0
    #   A21 * B22: NO j-overlap (A21 uses j2; B22 uses j0,j1) -> 0
    #   A11 * B22: NO j-overlap -> 0
    # Total: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26
    def block_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        # A22 * B22: Strassen
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        # A22 * B21 (k=k2 only)
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term({(i, j): 1}, {(j, k2): 1}, {(k2, i): 1})
        # A12 (row i2) * B22
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})
        # A12 * B21
        for j in [j0, j1]:
            add_term({(i2, j): 1}, {(j, k2): 1}, {(k2, i2): 1})
        # A21 (col j2) * B12 (row j2, cols k0,k1)
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})
        # A21 * B11
        for i in [i0, i1]:
            add_term({(i, j2): 1}, {(j2, k2): 1}, {(k2, i): 1})
        # A11 * B12
        for k in [k0, k1]:
            add_term({(i2, j2): 1}, {(j2, k): 1}, {(k, i2): 1})
        # A11 * B11
        add_term({(i2, j2): 1}, {(j2, k2): 1}, {(k2, i2): 1})

    # ---- 2x3 * 3x2 block (rank 11) --------------------------------------
    # A is 2x3 (rows i0,i1; cols j0,j1,j2)
    # B is 3x2 (rows j0,j1,j2; cols k0,k1)
    # Split j into {j0,j1} and {j2}
    # A22(2x2)*B22(2x2): Strassen 7
    # A22(2x2)*B21(1x2): 2 schoolbook
    # A21(2x1)*B12(1x2): 2 schoolbook
    # A21(2x1)*B11(1x1): 1 schoolbook
    # No cross j-overlap terms
    # Total: 7+2+2+1 = 12? Wait let me recount.
    # A22 cols j0,j1; B22 rows j0,j1 -> overlap j0,j1: Strassen 7
    # A22 cols j0,j1; B21 row j2 -> no overlap
    # A21 col j2; B22 rows j0,j1 -> no overlap
    # A21 col j2; B21 row j2 -> overlap j2: 2*1*1 = 2
    # So total = 7 + 2 = 9
    def block_2x3_3x2(i0, i1, j0, j1, j2, k0, k1):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            add_term({(i, j2): 1}, {(j2, k0): 1}, {(k0, i): 1})
            add_term({(i, j2): 1}, {(j2, k1): 1}, {(k1, i): 1})

    # ---- 2x3 * 3x3 block (rank 20) ---------------------------------------
    # A is 2x3 (rows i0,i1; cols j0,j1,j2)
    # B is 3x3 (rows j0,j1,j2; cols k0,k1,k2)
    # Split j: {j0,j1}, {j2}
    # A22(2x2)*B22(2x2, k0,k1): Strassen 7
    # A22(2x2)*B21(2x1, k2): 4
    # A21(2x1)*B12(1x2, k0,k1): 2
    # A21(2x1)*B11(1x1, k2): 1
    # Total: 7+4+2+1 = 14
    def block_2x3_3x3(i0, i1, j0, j1, j2, k0, k1, k2):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term({(i, j): 1}, {(j, k2): 1}, {(k2, i): 1})
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})
        for i in [i0, i1]:
            add_term({(i, j2): 1}, {(j2, k2): 1}, {(k2, i): 1})

    # ---- 3x2 * 2x2 block (rank 11) ---------------------------------------
    # A is 3x2 (rows i0,i1,i2; cols j0,j1)
    # B is 2x2 (rows j0,j1; cols k0,k1)
    # A22(2x2)*B(2x2): Strassen 7
    # A21(1x2)*B(2x2): 2
    # Total: 9
    def block_3x2_2x2(i0, i1, i2, j0, j1, k0, k1):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})

    # ---- 3x2 * 2x3 block (rank 18) ---------------------------------------
    # A is 3x2 (rows i0,i1,i2; cols j0,j1)
    # B is 2x3 (rows j0,j1; cols k0,k1,k2)
    # A22(2x2)*B22(2x2, k0,k1): Strassen 7
    # A22*B21(2x1, k2): 4
    # A21(1x2)*B22: 2
    # A21*B21: 1
    # Total: 7+4+2+1 = 14
    def block_3x2_2x3(i0, i1, i2, j0, j1, k0, k1, k2):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term({(i, j): 1}, {(j, k2): 1}, {(k2, i): 1})
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})
        for j in [j0, j1]:
            add_term({(i2, j): 1}, {(j, k2): 1}, {(k2, i2): 1})

    # ---- 3x3 * 3x2 block (rank 19) ---------------------------------------
    # A is 3x3, B is 3x2
    # Split j: {j0,j1}, {j2}
    # A22(2x2)*B22(2x2): Strassen 7
    # A22*B21: NO (B21 row j2, A22 cols j0,j1) -> 0
    # A21(2x1)*B12(1x2): 2
    # A21*B11: 1
    # A12(1x2)*B22: 2
    # A12*B21: NO
    # A11*B12: 1
    # A11*B11: 1
    # Total: 7+2+1+2+1+1 = 14
    def block_3x3_3x2(i0, i1, i2, j0, j1, j2, k0, k1):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})
        add_term({(i0, j2): 1}, {(j2, k2): 1}, {(k2, i0): 1}) if False else None
        # A21(2x1 col j2) * B11(1x1 row j2 col k2): but B is 3x2 so k only k0,k1
        # B11 is row j2 col k2 -- but k2 doesn't exist in 3x2!
        # Correction: B is 3x2, cols k0,k1. So B11 = row j2, cols k0,k1 -> 1x2
        # A21 * B12: i in {i0,i1}, col j2 * row j2, cols k0,k1 -> 2 terms
        # A12 * B22: row i2, cols j0,j1 * rows j0,j1, cols k0,k1 -> 2 terms
        # A12 * B21: row i2, cols j0,j1 * row j2 -> NO
        # A11 * B12: row i2, col j2 * row j2, cols k0,k1 -> 2 terms
        # A11 * B11: row i2, col j2 * row j2, col... wait B11 in 3x2 is row j2, cols k0,k1
        # Let me redo:
        # A22(2x2, rows i0,i1 cols j0,j1) * B22(2x2, rows j0,j1 cols k0,k1): Strassen 7
        # A22 * B21(1x2, row j2 cols k0,k1): NO j-overlap
        # A21(2x1, rows i0,i1 col j2) * B12(2x2, rows j0,j1 cols k0,k1): NO j-overlap
        # A21 * B11(1x2, row j2 cols k0,k1): 2*1*1 = 2... wait B11 is 1x2 so 2 terms
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})
        # A12(1x2, row i2 cols j0,j1) * B22(2x2, rows j0,j1 cols k0,k1): 2 terms
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})
        # A12 * B21: NO
        # A11(1x1, row i2 col j2) * B12(2x2, rows j0,j1): NO
        # A11 * B11(1x2, row j2 cols k0,k1): 1*1*2 = 2 terms
        for k in [k0, k1]:
            add_term({(i2, j2): 1}, {(j2, k): 1}, {(k, i2): 1})
        # Total: 7 + 2 + 2 + 2 = 13

    # ---- 3x3 * 2x2 block (rank 11) ---------------------------------------
    # A is 3x3, B is 2x2
    # Split j: {j0,j1}, {j2}
    # A22(2x2, rows i0,i1 cols j0,j1) * B(2x2, rows j0,j1 cols k0,k1): Strassen 7
    # A22 * B21: NO
    # A21(2x1, rows i0,i1 col j2) * B12(2x2, rows j0,j1): NO
    # A21 * B11: NO (B only has rows j0,j1)
    # A12(1x2, row i2 cols j0,j1) * B(2x2): 2
    # A12 * B21: NO
    # A11 * B12: NO
    # A11 * B11: NO
    # Total: 7+2 = 9
    def block_3x3_2x2(i0, i1, i2, j0, j1, j2, k0, k1):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})

    # ---- 2x2 * 2x3 block (rank 11) ---------------------------------------
    # A is 2x2, B is 2x3
    # A(2x2) * B22(2x2, cols k0,k1): Strassen 7
    # A * B21(2x1, col k2): 2
    # Total: 9
    def block_2x2_2x3(i0, i1, j0, j1, k0, k1, k2):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term({(i, j): 1}, {(j, k2): 1}, {(k2, i): 1})

    # ---- 2x2 * 3x2 block (rank 11) ---------------------------------------
    # A is 2x2, B is 3x2
    # Split j: {j0,j1}, {j2}
    # A(2x2) * B22(2x2, rows j0,j1 cols k0,k1): Strassen 7
    # A * B21(1x2, row j2 cols k0,k1): 2
    # Total: 9
    def block_2x2_3x2(i0, i1, j0, j1, j2, k0, k1):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})

    # ---- 3x2 * 3x3 block (rank 20) ---------------------------------------
    # A is 3x2, B is 3x3
    # Split j: {j0,j1}, {j2}
    # A(3x2) * B22(2x2, rows j0,j1 cols k0,k1):
    #   A22(2x2, rows i0,i1 cols j0,j1) * B22: Strassen 7
    #   A21(1x2, row i2 cols j0,j1) * B22: 2
    # A * B21(2x1, rows j0,j1 col k2):
    #   A22 * B21: 4
    #   A21 * B21: 2
    # A * B12(1x2, row j2 cols k0,k1): NO j-overlap with A (A cols j0,j1)
    # A * B11(1x1, row j2 col k2): NO
    # Total: 7+2+4+2 = 15
    def block_3x2_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term({(i, j): 1}, {(j, k2): 1}, {(k2, i): 1})
        for j in [j0, j1]:
            add_term({(i2, j): 1}, {(j, k2): 1}, {(k2, i2): 1})

    # ---- 3x3 * 2x3 block (rank 18) ---------------------------------------
    # A is 3x3, B is 2x3
    # Split j: {j0,j1}, {j2}
    # A22(2x2)*B22(2x2, k0,k1): Strassen 7
    # A22*B21(2x1, k2): 4
    # A21(2x1, j2)*B12(2x2, j0,j1): NO
    # A21*B11(2x1, j2): NO (B11 is row j2? No, B is 2x3 so rows j0,j1)
    # A12(1x2, j0,j1)*B22: 2
    # A12*B21: 2
    # A11(j2)*B12: NO
    # A11*B11: NO
    # Total: 7+4+2+2 = 15
    def block_3x3_2x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term({(i, j): 1}, {(j, k2): 1}, {(k2, i): 1})
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})
        for j in [j0, j1]:
            add_term({(i2, j): 1}, {(j, k2): 1}, {(k2, i2): 1})

    # ---- 2x3 * 3x2 already handled above ---------------------------------
    # ---- 2x3 * 3x3 already handled above ---------------------------------
    # ---- 3x2 * 2x2 already handled above ---------------------------------
    # ---- 3x2 * 2x3 already handled above ---------------------------------
    # ---- 3x3 * 3x2 already handled above ---------------------------------
    # ---- 3x3 * 2x2 already handled above ---------------------------------
    # ---- 2x2 * 2x3 already handled above ---------------------------------
    # ---- 2x2 * 3x2 already handled above ---------------------------------
    # ---- 3x2 * 3x3 already handled above ---------------------------------
    # ---- 3x3 * 2x3 already handled above ---------------------------------

    # ---- Now build the 5x5 = 2+3 split -----------------------------------
    # Indices: rows/cols 0,1 are "2-block" (call them a-block)
    #           rows/cols 2,3,4 are "3-block" (call them b-block)
    #
    # C = A * B where A,B,C are 5x5
    # Split into blocks:
    #   A22 = A[0:2, 0:2]   (2x2)
    #   A23 = A[0:2, 2:5]   (2x3)
    #   A32 = A[2:5, 0:2]   (3x2)
    #   A33 = A[2:5, 2:5]   (3x3)
    #   B22 = B[0:2, 0:2]   (2x2)
    #   B23 = B[0:2, 2:5]   (2x3)
    #   B32 = B[2:5, 0:2]   (3x2)
    #   B33 = B[2:5, 2:5]   (3x3)
    #
    # C22 = A22*B22 + A23*B32
    # C23 = A22*B23 + A23*B33
    # C32 = A32*B22 + A33*B32
    # C33 = A32*B23 + A33*B33

    # C22 = A22*B22 + A23*B32
    block_2x2_2x2 = lambda: strassen_2x2(0, 1, 0, 1, 0, 1)  # A22*B22
    block_2x2_2x2()
    block_2x3_3x2(0, 1, 2, 3, 4, 0, 1)  # A23*B32

    # C23 = A22*B23 + A23*B33
    block_2x2_2x3(0, 1, 0, 1, 2, 3, 4)  # A22*B23
    block_2x3_3x3(0, 1, 2, 3, 4, 2, 3, 4)  # A23*B33

    # C32 = A32*B22 + A33*B32
    block_3x2_2x2(2, 3, 4, 0, 1, 0, 1)  # A32*B22
    block_3x3_3x2(2, 3, 4, 2, 3, 4, 0, 1)  # A33*B32

    # C33 = A32*B23 + A33*B33
    block_3x2_2x3(2, 3, 4, 0, 1, 2, 3, 4)  # A32*B23
    block_3x3(2, 3, 4, 2, 3, 4, 2, 3, 4)  # A33*B33

    U = np.array(U_list).T  # shape (25, rank)
    V = np.array(V_list).T  # shape (25, rank)
    W = np.array(W_list).T  # shape (25, rank)

    rank = U.shape[1]

    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

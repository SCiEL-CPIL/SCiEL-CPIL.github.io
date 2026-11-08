import time
import numpy as np

_T0 = time.time()
_DEADLINE = 120.0

def _deadline_ok():
    return (time.time() - _T0) < (_DEADLINE - 8.0)

def construct():
    n, m, p = 3, 4, 6
    rank = 66
    
    # Pre-allocate lists for U, V, W to avoid object array overhead if needed,
    # but numpy arrays of floats are fast.
    # The interface expects a dict with lists or arrays?
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The current program returns .tolist(). I will do the same.
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    # Iterate over 3 blocks of 2 columns in B (and corresponding rows in C)
    for ko in range(0, p, 2):
        k1 = ko
        k2 = ko + 1
        
        # Part 1: Row 0 of A (1x4) * Block of B (4x2)
        # This is 1x4 * 4x2 -> 1x2
        # Rank 8: each of the 4 elements of A row 0 multiplies each of the 2 columns of B block
        for j in range(m):
            # Term for C[0, k1] += A[0, j] * B[j, k1]
            U[0 * m + j, r] = 1.0
            V[j * p + k1, r] = 1.0
            W[k1 * n + 0, r] = 1.0
            r += 1
            
            # Term for C[0, k2] += A[0, j] * B[j, k2]
            U[0 * m + j, r] = 1.0
            V[j * p + k2, r] = 1.0
            W[k2 * n + 0, r] = 1.0
            r += 1
            
        # Part 2: Rows 1-2 of A (2x4) * Block of B (4x2)
        # This is 2x4 * 4x2 -> 2x2
        # We use Strassen-like decomposition for 2x2 * 2x2 blocks.
        # Split A[1:3, :] into A_L (cols 0,1) and A_R (cols 2,3)
        # Split B_block into B_L (cols 0,1 of block, i.e. k1, k1+1? No, B is 4x2. 
        # Wait, B_block is 4x2. The columns are k1, k2.
        # The rows of B_block are 0,1,2,3.
        # We split the 4 rows of B into two groups of 2?
        # No, we split the 4 columns of A into two groups of 2.
        # A is 2x4. A_L = A[:, 0:2], A_R = A[:, 2:4].
        # B is 4x2. We cannot split B into 2x2 blocks horizontally because it only has 2 columns.
        # We must split B vertically?
        # B is 4x2. B_T = B[0:2, :], B_B = B[2:4, :].
        # Then A_L B_T + A_R B_B?
        # A_L is 2x2. B_T is 2x2. Product is 2x2.
        # A_R is 2x2. B_B is 2x2. Product is 2x2.
        # Sum is 2x2.
        # Rank of 2x2 * 2x2 is 7 (Strassen).
        # So we need 7 terms for A_L B_T and 7 terms for A_R B_B.
        # Total 14 terms.
        
        # Define indices for A_L, A_R, B_T, B_B
        # A rows: 1, 2. Cols: 0,1,2,3.
        # A_L cols: 0, 1. A_R cols: 2, 3.
        # B rows: 0, 1, 2, 3. Cols: k1, k2.
        # B_T rows: 0, 1. B_B rows: 2, 3.
        
        # Strassen for X(2x2) * Y(2x2) -> Z(2x2)
        # X = [[x11, x12], [x21, x22]]
        # Y = [[y11, y12], [y21, y22]]
        # m1 = (x11 + x22) * (y11 + y22)
        # m2 = (x21 + x22) * y11
        # m3 = x11 * (y12 - y22)
        # m4 = x22 * (y21 - y11)
        # m5 = (x11 + x12) * y22
        # m6 = (x21 - x11) * (y11 + y12)
        # m7 = (x12 - x22) * (y21 + y22)
        
        # We have two such multiplications:
        # 1. X = A_L (rows 1,2; cols 0,1), Y = B_T (rows 0,1; cols k1,k2)
        # 2. X = A_R (rows 1,2; cols 2,3), Y = B_B (rows 2,3; cols k1,k2)
        
        # Helper to add Strassen terms
        def add_strassen(U, V, W, r, 
                         x_rows, x_cols, # indices in A for X
                         y_rows, y_cols, # indices in B for Y
                         z_rows, z_cols): # indices in C for Z (output)
            
            # x_rows: [r1, r2] (global A rows)
            # x_cols: [c1, c2] (global A cols)
            # y_rows: [r3, r4] (global B rows)
            # y_cols: [c3, c4] (global B cols)
            # z_rows: [r5, r6] (global C rows)
            # z_cols: [c5, c6] (global C cols)
            
            # m1 = (x11 + x22) * (y11 + y22) -> z11 += m1, z12 += m1, z21 += m1, z22 += m1?
            # Strassen formulas for Z:
            # z11 = m1 - m2 + m3 + m4
            # z12 = m5 + m7
            # z21 = m2 + m6
            # z22 = m1 - m3 + m5 - m7
            
            # We need to map these to our U, V, W.
            
            # m1: U has x11, x22. V has y11, y22. W has z11, z12, z21, z22 with coeffs 1, 1, 1, 1?
            # No, W is fixed for the term.
            # The term is U_term * V_term * W_term.
            # U_term is a vector. V_term is a vector. W_term is a vector.
            
            # m1:
            # U: 1 at (x_rows[0], x_cols[0]) and (x_rows[1], x_cols[1])
            # V: 1 at (y_rows[0], y_cols[0]) and (y_rows[1], y_cols[1])
            # W: 1 at (z_rows[0], z_cols[0]) [z11]
            #     1 at (z_rows[0], z_cols[1]) [z12]
            #     1 at (z_rows[1], z_cols[0]) [z21]
            #     1 at (z_rows[1], z_cols[1]) [z22]
            # Wait, W is a vector in C space.
            # If W has 1 at multiple positions, it adds to multiple C entries.
            # This is valid.
            
            U[x_rows[0] * m + x_cols[0], r] += 1.0
            U[x_rows[1] * m + x_cols[1], r] += 1.0
            V[y_rows[0] * p + y_cols[0], r] += 1.0
            V[y_rows[1] * p + y_cols[1], r] += 1.0
            W[z_rows[0] * n + z_cols[0], r] += 1.0 # z11
            W[z_rows[0] * n + z_cols[1], r] += 1.0 # z12
            W[z_rows[1] * n + z_cols[0], r] += 1.0 # z21
            W[z_rows[1] * n + z_cols[1], r] += 1.0 # z22
            r += 1
            
            # m2: (x21 + x22) * y11
            # z21 += m2, z22 += m2
            U[x_rows[1] * m + x_cols[0], r] += 1.0 # x21
            U[x_rows[1] * m + x_cols[1], r] += 1.0 # x22
            V[y_rows[0] * p + y_cols[0], r] += 1.0 # y11
            W[z_rows[1] * n + z_cols[0], r] += 1.0 # z21
            W[z_rows[1] * n + z_cols[1], r] += 1.0 # z22
            r += 1
            
            # m3: x11 * (y12 - y22)
            # z11 += m3, z22 -= m3
            U[x_rows[0] * m + x_cols[0], r] += 1.0 # x11
            V[y_rows[0] * p + y_cols[1], r] += 1.0 # y12
            V[y_rows[1] * p + y_cols[1], r] -= 1.0 # -y22
            W[z_rows[0] * n + z_cols[0], r] += 1.0 # z11
            W[z_rows[1] * n + z_cols[1], r] -= 1.0 # -z22
            r += 1
            
            # m4: x22 * (y21 - y11)
            # z11 += m4, z21 += m4
            U[x_rows[1] * m + x_cols[1], r] += 1.0 # x22
            V[y_rows[1] * p + y_cols[0], r] += 1.0 # y21
            V[y_rows[0] * p + y_cols[0], r] -= 1.0 # -y11
            W[z_rows[0] * n + z_cols[0], r] += 1.0 # z11
            W[z_rows[1] * n + z_cols[0], r] += 1.0 # z21
            r += 1
            
            # m5: (x11 + x12) * y22
            # z12 += m5, z22 += m5
            U[x_rows[0] * m + x_cols[0], r] += 1.0 # x11
            U[x_rows[0] * m + x_cols[1], r] += 1.0 # x12
            V[y_rows[1] * p + y_cols[1], r] += 1.0 # y22
            W[z_rows[0] * n + z_cols[1], r] += 1.0 # z12
            W[z_rows[1] * n + z_cols[1], r] += 1.0 # z22
            r += 1
            
            # m6: (x21 - x11) * (y11 + y12)
            # z21 += m6
            U[x_rows[1] * m + x_cols[0], r] += 1.0 # x21
            U[x_rows[0] * m + x_cols[0], r] -= 1.0 # -x11
            V[y_rows[0] * p + y_cols[0], r] += 1.0 # y11
            V[y_rows[0] * p + y_cols[1], r] += 1.0 # y12
            W[z_rows[1] * n + z_cols[0], r] += 1.0 # z21
            r += 1
            
            # m7: (x12 - x22) * (y21 + y22)
            # z12 += m7, z22 -= m7
            U[x_rows[0] * m + x_cols[1], r] += 1.0 # x12
            U[x_rows[1] * m + x_cols[1], r] -= 1.0 # -x22
            V[y_rows[1] * p + y_cols[0], r] += 1.0 # y21
            V[y_rows[1] * p + y_cols[1], r] += 1.0 # y22
            W[z_rows[0] * n + z_cols[1], r] += 1.0 # z12
            W[z_rows[1] * n + z_cols[1], r] -= 1.0 # -z22
            r += 1
            
            return r

        # Apply Strassen for A_L * B_T
        # A_L: rows 1,2; cols 0,1
        # B_T: rows 0,1; cols k1,k2
        # Z: rows 1,2; cols k1,k2
        r = add_strassen(U, V, W, r,
                         [1, 2], [0, 1],
                         [0, 1], [k1, k2],
                         [1, 2], [k1, k2])
        
        # Apply Strassen for A_R * B_B
        # A_R: rows 1,2; cols 2,3
        # B_B: rows 2,3; cols k1,k2
        # Z: rows 1,2; cols k1,k2
        r = add_strassen(U, V, W, r,
                         [1, 2], [2, 3],
                         [2, 3], [k1, k2],
                         [1, 2], [k1, k2])

    if not _deadline_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(rank):
                    s += U[i * m + j, rr] * V[j * p + k, rr] * W[k * n + i, rr]
                if abs(s - 1.0) > 1e-9:
                    return _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

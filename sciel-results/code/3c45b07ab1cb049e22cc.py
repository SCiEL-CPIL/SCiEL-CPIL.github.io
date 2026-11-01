import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Block 1: A_1 (2x2) * B_1 (2x6)
    # A_1 uses A rows 0-1, cols 0-1. Indices in A: i*m+j
    # B_1 uses B rows 0-1, cols 0-5. Indices in B: j*p+k
    # Output C_1 is 2x6. Indices in W: k*n+i
    
    # Split B_1 into 3 blocks of 2 columns: k=0,1,2
    # Block k corresponds to columns 2k, 2k+1 of B (and C)
    
    for k in range(3):
        # B block indices: rows 0,1; cols 2k, 2k+1
        b_row_0 = 0
        b_row_1 = 1
        b_col_0 = 2 * k
        b_col_1 = 2 * k + 1
        
        # A block indices: rows 0,1; cols 0,1
        a_row_0 = 0
        a_row_1 = 1
        a_col_0 = 0
        a_col_1 = 1
        
        # Strassen for 2x2x2
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11) B00
        # M3 = A00 (B10 - B11)
        # M4 = A11 (B01 - B00)
        # M5 = (A00 + A01) B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Output C00 = M1 - M4 - M5 + M7
        # Output C10 = M3 + M4
        # Output C01 = M2 + M5
        # Output C11 = M1 - M2 + M3 + M6
        
        # We need to add terms to U, V, W.
        # The terms are linear combinations of A and B.
        # We can't just add "M1" as a single term because M1 is a product of sums.
        # We must expand them into the rank-1 terms.
        
        # M1 = (A00 + A11)(B00 + B11)
        # This requires 4 rank-1 terms if we don't combine?
        # No, Strassen works by defining new variables.
        # But the decomposition must be of the form sum U_c V_c W_c.
        # Each term c corresponds to one product of a linear combo of A and a linear combo of B.
        
        # So for M1, we have one term:
        # U part: A00 + A11 -> U[a_row_0, a_col_0] += 1, U[a_row_1, a_col_1] += 1
        # V part: B00 + B11 -> V[b_row_0, b_col_0] += 1, V[b_row_1, b_col_1] += 1
        # W part: C00 += 1, C11 += 1 (since M1 contributes to both)
        # Wait, M1 is a scalar. It contributes to C00 and C11.
        # So W[c0, i0] += 1, W[c1, i1] += 1?
        # No, W is (p*n, rank). W[k*n+i, c].
        # M1 contributes to C[0,0] and C[1,1]?
        # In Strassen:
        # C00 = M1 - M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 adds to C00 and C11.
        
        # Let's implement a helper for Strassen block
        def add_strassen(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, w_r0, w_r1, w_c0, w_c1):
            # w_r0, w_r1 are row indices in W (k)
            # w_c0, w_c1 are col indices in W (i)
            # Actually W index is k*n + i.
            # C00 is k=w_c0, i=w_r0? No.
            # C[i, k]. W[k*n + i, c].
            # So for C[0,0], W[0*n + 0] = W[0].
            # For C[1,1], W[1*n + 1] = W[3] (if n=2).
            
            # M1
            U[a_r0*m + a_c0][col] = 1
            U[a_r1*m + a_c1][col] = 1
            V[b_r0*p + b_c0][col] = 1
            V[b_r1*p + b_c1][col] = 1
            W[w_c0*n + w_r0][col] = 1 # C00
            W[w_c1*n + w_r1][col] = 1 # C11
            col += 1
            
            # M2
            U[a_r1*m + a_c0][col] = 1
            U[a_r1*m + a_c1][col] = 1
            V[b_r0*p + b_c0][col] = 1
            W[w_c1*n + w_r0][col] = 1 # C01
            col += 1
            
            # M3
            U[a_r0*m + a_c0][col] = 1
            V[b_r1*p + b_c0][col] = 1
            V[b_r1*p + b_c1][col] = -1
            W[w_c0*n + w_r1][col] = 1 # C10
            col += 1
            
            # M4
            U[a_r1*m + a_c1][col] = 1
            V[b_r0*p + b_c1][col] = 1
            V[b_r0*p + b_c0][col] = -1
            W[w_c0*n + w_r0][col] = -1 # C00
            col += 1
            
            # M5
            U[a_r0*m + a_c0][col] = 1
            U[a_r0*m + a_c1][col] = 1
            V[b_r1*p + b_c1][col] = 1
            W[w_c0*n + w_r0][col] = -1 # C00
            W[w_c1*n + w_r0][col] = 1 # C01
            col += 1
            
            # M6
            U[a_r1*m + a_c0][col] = 1
            U[a_r0*m + a_c0][col] = -1
            V[b_r0*p + b_c0][col] = 1
            V[b_r0*p + b_c1][col] = 1
            W[w_c1*n + w_r1][col] = 1 # C11
            col += 1
            
            # M7
            U[a_r0*m + a_c1][col] = 1
            U[a_r1*m + a_c1][col] = -1
            V[b_r1*p + b_c0][col] = 1
            V[b_r1*p + b_c1][col] = 1
            W[w_c0*n + w_r0][col] = 1 # C00
            col += 1

        add_strassen(0, 1, 0, 1, 0, 1, b_col_0, b_col_1, 0, 1, b_col_0, b_col_1)

    # Block 2: A_2 (2x3) * B_2 (3x6)
    # A_2 uses A rows 0-1, cols 2-4.
    # B_2 uses B rows 2-4, cols 0-5.
    
    for k in range(3):
        # B block indices: rows 2,3,4; cols 2k, 2k+1
        b_r0 = 2
        b_r1 = 3
        b_r2 = 4
        b_c0 = 2 * k
        b_c1 = 2 * k + 1
        
        # A block indices: rows 0,1; cols 2,3,4
        a_r0 = 0
        a_r1 = 1
        a_c0 = 2
        a_c1 = 3
        a_c2 = 4
        
        # Decomposition:
        # A2 = [A20 (2x2), A21 (2x1)]
        # B2k = [B2k0 (2x2); B2k1 (1x2)]
        # C2k = A20 B2k0 + A21 B2k1
        
        # Strassen for A20 B2k0
        # A20 uses cols 2,3. B2k0 uses rows 2,3.
        add_strassen(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, 0, 1, b_c0, b_c1)
        
        # Outer product for A21 B2k1
        # A21 is 2x1 (col 4). B2k1 is 1x2 (row 4).
        # C2k += A21 * B2k1
        # C[0,0] += A[0,4] * B[4, 2k]
        # C[1,0] += A[1,4] * B[4, 2k]
        # C[0,1] += A[0,4] * B[4, 2k+1]
        # C[1,1] += A[1,4] * B[4, 2k+1]
        
        # 4 terms
        U[a_r0*m + a_c2][col] = 1
        V[b_r2*p + b_c0][col] = 1
        W[b_c0*n + a_r0][col] = 1
        col += 1
        
        U[a_r1*m + a_c2][col] = 1
        V[b_r2*p + b_c0][col] = 1
        W[b_c0*n + a_r1][col] = 1
        col += 1
        
        U[a_r0*m + a_c2][col] = 1
        V[b_r2*p + b_c1][col] = 1
        W[b_c1*n + a_r0][col] = 1
        col += 1
        
        U[a_r1*m + a_c2][col] = 1
        V[b_r2*p + b_c1][col] = 1
        W[b_c1*n + a_r1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

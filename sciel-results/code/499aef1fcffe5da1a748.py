import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Split: first 2 columns/rows, last 3
    s = 2  # block size for first part
    
    # A is n×m = 3×5, B is m×p = 5×5, C is p×n = 5×3
    # Block decomposition:
    # A = [A11 (n×s) | A12 (n×(m-s))]
    # B = [[B11 (s×s), B12 (s×(p-s))],
    #      [B21 ((m-s)×s), B22 ((m-s)×(p-s))]]
    # C = [C11 (s×n) | C12 ((p-s)×n)]
    #      [C21 (s×n) | C22 ((p-s)×n)]
    
    # Strassen for block matrix multiplication:
    # C11 = M1 + M4 - M7 + M6
    # C12 = M5 + M7
    # C21 = M3 + M4
    # C22 = M1 - M4 + M5 + M6
    # where:
    # M1 = (A11+A22)(B22-B12)
    # M2 = (A11+A22)B12  [not needed in standard Strassen for 2x2, skip]
    # Actually standard Strassen:
    # M1 = (A11+A22)(B12-B22)
    # M2 = (A21+A22)B11
    # M3 = (A11+A12)B22
    # M4 = (A21-A11)(B11+B12)
    # M5 = (A12-A22)(B21+B22)
    # M6 = A11B11
    # M7 = A22B22
    # C11 = M6 + M4 - M7 + M5
    # C12 = M3 + M7
    # C21 = M6 + M1
    # C22 = M2 - M1 - M4 + M3
    
    # Wait, let me use the standard Strassen formulas correctly:
    # For C = A*B where A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # M1 = (A11+A22)(B22-B12)
    # M2 = (A21+A22)B11
    # M3 = (A11+A12)B22
    # M4 = (A21-A11)(B11+B12)
    # M5 = (A12-A22)(B21+B22)
    # M6 = A11B11
    # M7 = A22B22
    # C11 = M1 + M4 - M5 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Each Mi is a matrix product. We decompose each into rank-1 terms.
    
    # A11: n×s = 3×2, A12: n×(m-s) = 3×3
    # A21: n×s = 3×2, A22: n×(m-s) = 3×3
    # B11: s×s = 2×2, B12: s×(p-s) = 2×3
    # B21: (m-s)×s = 3×2, B22: (m-s)×(p-s) = 3×3
    
    # Rank of each Mi (schoolbook):
    # M1 = (A11+A22)(B22-B12): 3×3 * 3×3 = 9
    # M2 = (A21+A22)B11: 3×3 * 2×2 = 6
    # M3 = (A11+A12)B22: 3×3 * 3×3 = 9
    # M4 = (A21-A11)(B11+B12): 3×2 * 2×3 = 6
    # M5 = (A12-A22)(B21+B22): 3×3 * 3×3 = 9
    # M6 = A11B11: 3×2 * 2×2 = 4
    # M7 = A22B22: 3×3 * 3×3 = 9
    # Total: 9+6+9+6+9+4+9 = 52
    
    # C is p×n = 5×3, indexed as C[k*n+i, r] = C[k][i]
    
    # Build U, V, W for each rank-1 term
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec has length n*m, v_vec has length m*p, w_vec has length p*n."""
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    def schoolbook_decompose(A_row_indices, B_row_indices, B_col_indices, 
                              A_coeff, B_coeff, 
                              C_row_indices, C_col_indices, C_coeff):
        """Add rank-1 terms for sum of A_coeff*A[i,j]*B_coeff*B[k,l] contributing to C_coeff*C[c_r,c_c]
        A is indexed by (i,j) where i in A_row_indices (0..n-1), j in 0..m-1
        B is indexed by (k,l) where k in 0..m-1, l in 0..p-1
        C is indexed by (c_r, c_c) where c_r in 0..p-1, c_c in 0..n-1
        W index: c_r*n + c_c
        U index: i*m + j
        V index: k*p + l
        """
        for i in A_row_indices:
            for j in B_row_indices:  # j is the column of A = row of B
                if A_coeff[i][j] == 0 or B_coeff[j][l if False else 0] == 0:
                    pass
        # Simpler: iterate over all pairs
        for i in range(n):
            for j in range(m):
                a_val = A_coeff[i][j]
                for k in range(m):
                    for l in range(p):
                        b_val = B_coeff[k][l]
                        if a_val == 0 or b_val == 0:
                            continue
                        # Contribution to C[c_r][c_c] for c_r in C_row_indices, c_c in C_col_indices
                        # C[c_r][c_c] gets a_val * b_val * C_coeff[c_r][c_c]
                        # But this is only if j == k (A's column j matches B's row k)
                        if j != k:
                            continue
                        for cr in C_row_indices:
                            for cc in C_col_indices:
                                c_val = C_coeff[cr][cc]
                                if c_val == 0:
                                    continue
                                # Term: a_val * b_val * c_val * A[i,j] * B[j,l] -> C[l, cc]
                                # Wait, C_row_indices should be {l} for this to make sense
                                # Let me reconsider.
                                pass
    
    # Let me take a more direct approach: enumerate all rank-1 terms explicitly.
    
    # For a product of matrices X (r1×c1) * Y (c1×c2) = Z (r1×c2),
    # the schoolbook decomposition has c1 terms: for each t in range(c1):
    #   U: e_{i} * X[i,t] for each i
    #   V: e_{t} * Y[t,j] for each j  
    #   W: e_{i,j} * 1
    # Actually: sum_t (col_t of X)(row_t of Y) where col_t of X is vector in R^{r1}
    # and row_t of Y is vector in R^{c2}, outer product gives r1×c2 matrix.
    # So U vector = X[:,t] (length r1), V vector = Y[t,:] (length c2), W vector = ones (length r1*c2)
    # But we need to embed these in the full n×m, m×p, p×n spaces.
    
    # Let me define helper to add a matrix product decomposition.
    # Product: X (rows from A-space, cols from m-space) * Y (rows from m-space, cols from p-space)
    # X is indexed by A rows (subset of 0..n-1) and A cols (subset of 0..m-1)
    # Y is indexed by B rows (subset of 0..m-1) and B cols (subset of 0..p-1)
    # Result Z indexed by C rows (subset of 0..p-1) and C cols (subset of 0..n-1)
    
    def add_matrix_product(X_rows, X_cols, X_coeffs, 
                           Y_rows, Y_cols, Y_coeffs,
                           Z_rows, Z_cols, Z_coeffs,
                           scale_u=1.0, scale_v=1.0, scale_w=1.0):
        """
        Add rank-1 terms for the matrix product.
        X is |X_rows| × |X_cols| with entries X_coeffs[i][j] * A[X_rows[i], X_cols[j]]
        Y is |Y_rows| × |Y_cols| with entries Y_coeffs[i][j] * B[Y_rows[i], Y_cols[j]]
        Z is |Z_rows| × |Z_cols| with entries Z_coeffs[i][j] * C[Z_rows[i], Z_cols[j]]
        
        The product X*Y has the same "inner dimension" which must match.
        X_cols and Y_rows should both be subsets of 0..m-1 and represent the same indices.
        
        For each inner index t (which is in both X_cols and Y_rows):
          U vector: for each (i, col) in X: if col == t, add X_coeffs[i][col] * scale_u to position (X_rows[i], col)
          V vector: for each (row, j) in Y: if row == t, add Y_coeffs[row][j] * scale_v to position (row, Y_cols[j])
          W vector: for each (i, j) in Z: add Z_coeffs[i][j] * scale_w to position (Z_rows[i], Z_cols[j])
        """
        # Find the inner dimension indices
        inner_indices = list(X_cols)
        
        for t_idx, t in enumerate(inner_indices):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            # Fill U: X[:, t_idx]
            for i in range(len(X_rows)):
                if X_coeffs[i][t_idx] != 0:
                    r = X_rows[i]
                    c = X_cols[t_idx]
                    u_vec[r * m + c] = X_coeffs[i][t_idx] * scale_u
            
            # Fill V: Y[t_idx, :]
            for j in range(len(Y_cols)):
                if Y_coeffs[t_idx][j] != 0:
                    r = Y_rows[t_idx]
                    c = Y_cols[j]
                    v_vec[r * p + c] = Y_coeffs[t_idx][j] * scale_v
            
            # Fill W: all of Z
            for i in range(len(Z_rows)):
                for j in range(len(Z_cols)):
                    if Z_coeffs[i][j] != 0:
                        r = Z_rows[i]
                        c = Z_cols[j]
                        w_vec[r * n + c] = Z_coeffs[i][j] * scale_w
            
            # Only add if non-zero
            if any(x != 0 for x in u_vec) and any(x != 0 for x in v_vec) and any(x != 0 for x in w_vec):
                add_term(u_vec, v_vec, w_vec)
    
    # Define block indices
    # A rows: 0,1,2 (n=3)
    # A cols / B rows: 0,1,2,3,4 (m=5)
    # B cols / C rows: 0,1,2,3,4 (p=5)
    # C cols: 0,1,2 (n=3)
    
    # A11: rows 0-2, cols 0-1
    # A12: rows 0-2, cols 2-4
    # A21: rows 0-2, cols 0-1 (same as A11 in terms of A, but it's the lower block)
    # Wait, I'm confusing myself. A is 3×5. There's no "A21" and "A22" in the standard sense.
    
    # Let me restart with clearer notation.
    # A (3×5) split into columns: A_L (3×2) = A[:, 0:2], A_R (3×3) = A[:, 2:5]
    # B (5×5) split into rows and columns:
    #   B_TL (2×2) = B[0:2, 0:2]
    #   B_TR (2×3) = B[0:2, 2:5]
    #   B_BL (3×2) = B[2:5, 0:2]
    #   B_BR (3×3) = B[2:5, 2:5]
    # C (5×3) split into rows:
    #   C_T (2×3) = C[0:2, :]
    #   C_B (3×3) = C[2:5, :]
    
    # C = [A_L B_TL + A_R B_BL | A_L B_TR + A_R B_BR]
    #     [A_L B_TL + A_R B_BL | A_L B_TR + A_R B_BR]  -- no wait
    # C[0:2, :] = A_L * B_TL + A_R * B_BL   (2×3)
    # C[2:5, :] = A_L * B_TR + A_R * B_BR   (3×3)
    
    # Using Strassen on the "block" structure where the blocks are:
    # X = A_L (3×2), Y = A_R (3×3), U = B_TL (2×2), V = B_TR (2×3), W = B_BL (3×2), Z = B_BR (3×3)
    # C_top = X*U + Y*W  (3×3 result, but only first 2 rows of C)
    # C_bot = X*V + Y*Z  (3×3 result, last 3 rows of C)
    
    # This isn't a standard 2×2 block multiplication. Let me think differently.
    
    # Actually, the matrix product is:
    # C[i,k] = sum_j A[i,j] B[j,k] for i in 0..2, k in 0..4
    # 
    # Split j into j in {0,1} and j in {2,3,4}:
    # C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{4} A[i,j] B[j,k]
    #
    # For each i, define:
    # a_L[i] = (A[i,0], A[i,1]) -- 1×2 vector
    # a_R[i] = (A[i,2], A[i,3], A[i,4]) -- 1×3 vector
    #
    # C[i,k] for k in {0,1}: a_L[i] · B[0:2, k]
    # C[i,k] for k in {2,3,4}: a_L[i] · B[0:2, k] + a_R[i] · B[2:5, k]
    
    # This is getting complicated. Let me just do schoolbook but with a smarter grouping.
    
    # Alternative: Use the fact that we can group terms.
    # Schoolbook has 3*5*5 = 75 terms.
    # Can we find linear dependencies to reduce?
    
    # Let me just implement the Strassen-style decomposition properly.
    
    # Treat the 5×5 multiplication as a 2×2 block multiplication where:
    # A_block = [[A11, A12], [A21, A22]] but A is 3×5, so this doesn't directly work.
    
    # New approach: Since n=3, each row of A produces a row of C.
    # For row i of A (length 5) times B (5×5) = row of C (length 5).
    # This is a 1×5 * 5×5 = 1×5 matrix product.
    # Rank of this is 5 (schoolbook).
    # Can we do better? The tensor for 1×5 * 5×5 * 1×5 has rank 5.
    # So for 3 independent rows, we need at least 3*5 = 15, but they share the same B.
    
    # Actually, the full tensor is the 3×5×5 matrix multiplication tensor.
    # The standard rank is 3*5*5 = 75.
    # Using Strassen-like ideas on the 5×5 part: if we could decompose 5×5*5×5 with rank < 25,
    # we could get 3 * (rank of 5×5) total.
    
    # Best known upper bound for 5×5 is 19 (Landsberg, 1994, or earlier).
    # But I don't have the explicit decomposition memorized correctly.
    
    # Let me try a different approach: use a 2×2 Strassen on overlapping submatrices.
    
    # Split 5 = 2+2+1. Use Strassen on the 2×2 parts.
    
    # Actually, let me just implement a valid rank-52 decomposition using the block Strassen idea.
    
    # Clear approach:
    # C = A * B where A is 3×5, B is 5×5
    # Split columns of A and rows/cols of B at position 2:
    # A = [A1 | A2] where A1 is 3×2, A2 is 3×3
    # B = [[B11, B12], [B21, B22]] where B11 is 2×2, B12 is 2×3, B21 is 3×2, B22 is 3×3
    # C = [A1*B11 + A2*B21 | A1*B12 + A2*B22]  (5×3 matrix, split into top 2 rows and bottom 3 rows)
    # C_top (2×3) = A1*B11 + A2*B21
    # C_bot (3×3) = A1*B12 + A2*B22
    
    # Use Strassen on this "block" structure:
    # Define:
    # P1 = (A1 + A2) * (B22 - B12)  -- 3×3 * 3×3 = 9 terms
    # P2 = (A2) * (B11 + B12)       -- 3×3 * 2×3 = 6 terms  
    # P3 = (A1) * (B21 - B11)       -- 3×2 * 2×2 = 4 terms
    # P4 = (A1 + A2) * B22          -- 3×3 * 3×3 = 9 terms
    # P5 = (A2 - A1) * (B11 + B21)  -- 3×3 * 2×2... wait dimensions don't work
    
    # I'm making errors with dimensions. Let me be very careful.
    
    # A1: 3×2, A2: 3×3
    # B11: 2×2, B12: 2×3, B21: 3×2, B22: 3×3
    
    # C_top (2×3) = A1*B11 (3×2 * 2×2 = 3×2) + A2*B21 (3×3 * 3×2 = 3×2)
    #   Wait, C is 5×3, so C_top is first 2 rows: 2×3
    #   A1*B11: (3×2)*(2×2) = 3×2, but we only want first 2 rows -> 2×2
    #   This doesn't work cleanly.
    
    # I think the issue is that A has 3 rows but the block split gives 3×2 and 3×3,
    # while B's blocks give results with 3 rows, but C only has 5 rows total.
    
    # Let me abandon the block Strassen approach and just do a careful schoolbook
    # implementation that I can verify is correct, then see if I can find any
    # linear dependencies.
    
    # Actually, let me try yet another approach: use Kronecker product.
    # 5×5 = (2×2) ⊗ something? No, 5 is prime.
    
    # Let me just implement a correct rank-75 decomposition and verify it works.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

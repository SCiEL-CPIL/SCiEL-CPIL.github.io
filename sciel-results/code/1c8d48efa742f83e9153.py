import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_column(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows, a_cols, b_rows, b_cols, c_rows, c_cols are lists of 2 indices
        # Local indices:
        # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        # Map local to global index
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        a_idx = [
            a_rows[0]*m + a_cols[0], # A00
            a_rows[0]*m + a_cols[1], # A01
            a_rows[1]*m + a_cols[0], # A10
            a_rows[1]*m + a_cols[1]  # A11
        ]
        b_idx = [
            b_rows[0]*p + b_cols[0], # B00
            b_rows[0]*p + b_cols[1], # B01
            b_rows[1]*p + b_cols[0], # B10
            b_rows[1]*p + b_cols[1]  # B11
        ]
        c_idx = [
            c_cols[0]*n + c_rows[0], # C00
            c_cols[1]*n + c_rows[0], # C01
            c_cols[0]*n + c_rows[1], # C10
            c_cols[1]*n + c_rows[1]  # C11
        ]
        
        # Strassen terms
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # Coefficients for U (A)
        # 7 terms, 4 A entries each
        u_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0],# M6
            [0, 1, 0, -1] # M7
        ]
        
        # Coefficients for V (B)
        v_coeffs = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1],# M3
            [0, 0, 1, -1],# M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        # Coefficients for W (C)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        w_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, -1],# M2
            [0, 1, 0, 1], # M3
            [1, 0, 1, 0], # M4
            [-1, 1, 0, 0],# M5
            [0, 0, 0, 1], # M6
            [1, 0, 0, 0]  # M7
        ]
        
        for l in range(7):
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            for a_l in range(4):
                if u_coeffs[l][a_l] != 0:
                    u_vec[a_idx[a_l]] = u_coeffs[l][a_l]
            
            for b_l in range(4):
                if v_coeffs[l][b_l] != 0:
                    v_vec[b_idx[b_l]] = v_coeffs[l][b_l]
            
            for c_l in range(4):
                if w_coeffs[l][c_l] != 0:
                    w_vec[c_idx[c_l]] = w_coeffs[l][c_l]
            
            add_column(u_vec, v_vec, w_vec)
            
    def add_dot(a_row, a_cols, b_cols, c_row):
        # Computes C[c_row, c_cols] = A[a_row, a_cols] dot B[:, b_cols]
        # a_cols: list of m indices
        # b_cols: list of p indices (should be same length as a_cols)
        # This is a 1 x m x p multiplication
        # Rank is p (if m=p) or min(m,p)? No, it's m dot products?
        # No, C is 1 x p.
        # C_k = sum_j A_j B_jk.
        # This can be done with p terms?
        # Term k: A * B_{:,k}.
        # U: A entries. V: B_{:,k} entries. W: C_k entry.
        # We can just use p terms, each picking one column of B.
        # Or m terms, each picking one row of A?
        # Standard basis:
        # For each j in a_cols:
        #   U: A[a_row, j] = 1
        #   V: B[j, b_cols] = 1 (vector)
        #   W: C[c_row, b_cols] = 1 (vector)
        # This uses m terms.
        # If we use p terms:
        # For each k in b_cols:
        #   U: A[a_row, a_cols] = 1 (vector)
        #   V: B[a_cols, k] = 1 (vector)
        #   W: C[c_row, k] = 1
        
        # Let's use p terms (one per column of output)
        for k_idx, b_col in enumerate(b_cols):
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            for a_col in a_cols:
                u_vec[a_row*m + a_col] = 1
                v_vec[a_col*p + b_col] = 1
                # W index for C[c_row, b_col] is b_col*n + c_row
                w_vec[b_col*n + c_row] = 1
            add_column(u_vec, v_vec, w_vec)

    # Part 1: 3x4x4 (Columns 0-3 of B)
    # Split A into rows 0-1 and 2
    # Sub 1a: 2x4x4 (A rows 0-1, B cols 0-3)
    # Split A rows 0-1 into 2x2 blocks?
    # A is 2x4. Cols 0-1 and 2-3.
    # B is 4x4. Rows 0-1, 2-3. Cols 0-1, 2-3.
    # We need 4 Strassen calls for the 4 blocks of the result?
    # Result is 2x4.
    # C[0:2, 0:2] comes from A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    # This is not just 4 independent Strassen.
    # My previous logic:
    # AB = [A11 B11 + A12 B21, A11 B12 + A12 B22]
    # This requires 4 products:
    # P1 = A11 B11
    # P2 = A12 B21
    # P3 = A11 B12
    # P4 = A12 B22
    # Each is 2x2x2.
    # We can compute each with Strassen.
    # Total 28 terms.
    
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # C11 (result rows 0-1, cols 0-1): P1 + P2
    # C12 (result rows 0-1, cols 2-3): P3 + P4
    
    # We need to add the Strassen decompositions for P1, P2, P3, P4.
    # But the W vectors must be summed correctly.
    # P1 contributes to C[0:2, 0:2]
    # P2 contributes to C[0:2, 0:2]
    # P3 contributes to C[0:2, 2:4]
    # P4 contributes to C[0:2, 2:4]
    
    # Strassen adds 7 columns for each product.
    # We call add_strassen 4 times.
    # The W vectors from these calls will be added to the global W.
    # Since we are just appending columns, the linearity handles the sum.
    # C = Sum of terms.
    # So we just need to ensure the W vectors point to the correct global C indices.
    
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]) # P1 -> C[0:2, 0:2]
    add_strassen([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1]) # P2 -> C[0:2, 0:2]
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3]) # P3 -> C[0:2, 2:4]
    add_strassen([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3]) # P4 -> C[0:2, 2:4]
    
    # Sub 1b: 1x4x4 (A row 2, B cols 0-3)
    add_dot(2, [0, 1, 2, 3], [0, 1, 2, 3], 2)
    
    # Part 2: 3x4x2 (Columns 4-5 of B)
    # Sub 2a: 2x4x2 (A rows 0-1, B cols 4-5)
    # AB = A11 B1 + A12 B2
    # B is 4x2. Split rows into 0-1 and 2-3.
    # B1: rows 0-1, cols 4-5
    # B2: rows 2-3, cols 4-5
    # P5 = A11 B1
    # P6 = A12 B2
    # Result C[0:2, 4:6] = P5 + P6
    
    add_strassen([0, 1], [0, 1], [0, 1], [4, 5], [0, 1], [4, 5]) # P5 -> C[0:2, 4:6]
    add_strassen([0, 1], [2, 3], [2, 3], [4, 5], [0, 1], [4, 5]) # P6 -> C[0:2, 4:6]
    
    # Sub 2b: 1x4x2 (A row 2, B cols 4-5)
    add_dot(2, [0, 1, 2, 3], [4, 5], 2)
    
    # Total rank: 4*7 + 4 + 2*7 + 2 = 28 + 4 + 14 + 2 = 48
    
    U = np.array(U_cols).T # Shape (n*m, rank)
    V = np.array(V_cols).T # Shape (m*p, rank)
    W = np.array(W_cols).T # Shape (p*n, rank)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

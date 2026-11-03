import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = 2, 4, 4
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (8, rank)
    
    # We'll build lists of (u_idx, u_val, v_idx, v_val, w_idx, w_val) for each rank-1 term
    terms = []
    
    # Strassen's 7 products for multiplying 2x2 X by 2x2 Y
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01-x11)*(y10+y11)
    # Result:
    # R00 = M1+M4-M5+M7
    # R01 = M3+M5
    # R10 = M2+M4
    # R11 = M1-M2+M3+M6
    
    # For each 2x2 multiplication, we need to specify:
    # - The linear form in A (which entries of A with what coefficients)
    # - The linear form in B (which entries of B with what coefficients)
    # - How the result M contributes to output entries
    
    # Block structure:
    # A is 2x4: A = [A1 | A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B is 4x4: B = [[B11, B12], [B21, B22]] each 2x2
    # C is 2x4: C = [C1 | C2] where C1 = C[:, 0:2], C2 = C[:, 2:4]
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # For a 2x2 multiplication X*Y where X = A[rows, cols_a] and Y = B[cols_b, cols_c]:
    # X entry (r, c) corresponds to A[r, cols_a + c]
    # Y entry (r, c) corresponds to B[cols_b + r, cols_c + c]
    # Result entry (r, c) corresponds to C[r, cols_c + c]
    
    def add_strassen_mult(A_rows, A_cols, B_rows, B_cols, C_cols):
        """
        Add Strassen decomposition for A[A_rows, A_cols] * B[B_rows, B_cols] -> C[:, C_cols]
        A_rows: list of 2 row indices in A
        A_cols: list of 2 col indices in A
        B_rows: list of 2 row indices in B
        B_cols: list of 2 col indices in B
        C_cols: list of 2 col indices in C
        """
        # X[r,c] = A[A_rows[r], A_cols[c]]
        # Y[r,c] = B[B_rows[r], B_cols[c]]
        # R[r,c] = C[A_rows[r], C_cols[c]]
        
        # For each of the 7 Strassen products:
        # Each product is (linear form in X) * (linear form in Y)
        # Linear form in X: sum of coeff * X[r,c] = sum of coeff * A[A_rows[r], A_cols[c]]
        # Linear form in Y: sum of coeff * Y[r,c] = sum of coeff * B[B_rows[r], B_cols[c]]
        # Contribution to R: sum of coeff * R[r,c] = sum of coeff * C[A_rows[r], C_cols[c]]
        
        # Define the 7 Strassen products
        # Each: (X_coeffs, Y_coeffs, R_coeffs)
        # X_coeffs: dict {(r,c): coeff}
        # Y_coeffs: dict {(r,c): coeff}
        # R_coeffs: dict {(r,c): coeff}
        
        strassen_products = [
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            # M3
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
            # M4
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
        ]
        
        for x_coeffs, y_coeffs, r_coeffs in strassen_products:
            # Build U, V, W contributions
            # U: for each (r,c) in x_coeffs, A[A_rows[r], A_cols[c]] has coeff
            # V: for each (r,c) in y_coeffs, B[B_rows[r], B_cols[c]] has coeff
            # W: for each (r,c) in r_coeffs, C[A_rows[r], C_cols[c]] has coeff
            
            u_entries = {}
            v_entries = {}
            w_entries = {}
            
            for (r, c), coeff in x_coeffs.items():
                a_row = A_rows[r]
                a_col = A_cols[c]
                u_idx = a_row * m + a_col
                u_entries[u_idx] = u_entries.get(u_idx, 0) + coeff
            
            for (r, c), coeff in y_coeffs.items():
                b_row = B_rows[r]
                b_col = B_cols[c]
                v_idx = b_row * p + b_col
                v_entries[v_idx] = v_entries.get(v_idx, 0) + coeff
            
            for (r, c), coeff in r_coeffs.items():
                c_row = A_rows[r]
                c_col = C_cols[c]
                w_idx = c_col * n + c_row
                w_entries[w_idx] = w_entries.get(w_idx, 0) + coeff
            
            terms.append((u_entries, v_entries, w_entries))
    
    # C1 = A1*B11 + A2*B21
    # A1 = A[:, 0:2], B11 = B[0:2, 0:2], C1 = C[:, 0:2]
    add_strassen_mult([0,1], [0,1], [0,1], [0,1], [0,1])  # A1*B11
    add_strassen_mult([0,1], [2,3], [2,3], [0,1], [0,1])  # A2*B21
    
    # C2 = A1*B12 + A2*B22
    # A1 = A[:, 0:2], B12 = B[0:2, 2:4], C2 = C[:, 2:4]
    add_strassen_mult([0,1], [0,1], [0,1], [2,3], [2,3])  # A1*B12
    add_strassen_mult([0,1], [2,3], [2,3], [2,3], [2,3])  # A2*B22
    
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries.items():
            U[idx][c] = val
        for idx, val in v_entries.items():
            V[idx][c] = val
        for idx, val in w_entries.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}

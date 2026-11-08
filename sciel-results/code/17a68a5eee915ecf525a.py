import numpy as np

def construct():
    # Dimensions: A is 2x4, B is 4x4, C is 2x4
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen algorithm for 2x2 matrix multiplication
    # P = A * B, where A, B are 2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # P = [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    
    # We decompose the 2x4 * 4x4 multiplication into 4 independent 2x2 * 2x2 multiplications
    # A is split into A_left (2x2) and A_right (2x2)
    # B is split into B_top_left (2x2), B_bottom_left (2x2), B_top_right (2x2), B_bottom_right (2x2)
    # C is split into C_left (2x2) and C_right (2x2)
    
    # C_left = A_left * B_top_left + A_right * B_bottom_left
    # C_right = A_left * B_top_right + A_right * B_bottom_right
    
    # Since Strassen is bilinear, we can compute each product separately and add the results.
    # However, rank is additive for independent components.
    # Rank(A_left * B_top_left) = 7
    # Rank(A_right * B_bottom_left) = 7
    # Rank(A_left * B_top_right) = 7
    # Rank(A_right * B_bottom_right) = 7
    # Total Rank = 28
    
    # Helper to apply Strassen to specific blocks
    def apply_strassen_block(base_rank, a_col_start, b_row_start, b_col_start, c_col_start):
        """
        Applies Strassen decomposition for a 2x2 * 2x2 product.
        A_block is at columns a_col_start..a_col_start+1 of the original A (2x4)
        B_block is at rows b_row_start..b_row_start+1, cols b_col_start..b_col_start+1 of original B (4x4)
        C_block is at columns c_col_start..c_col_start+1 of the original C (2x4)
        
        Indices in U, V, W correspond to flattened (row, col) indices.
        U index: (r, c) -> r*m + c. Here r in [0,1], c in [0,3].
        V index: (r, c) -> r*p + c. Here r in [0,3], c in [0,3].
        W index: (k, i) -> k*n + i. Here k is col in [0,3], i is row in [0,1].
        """
        
        # Strassen formulas:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # P00 = M1 + M4 - M5 + M7
        # P01 = M3 + M5
        # P10 = M2 + M4
        # P11 = M1 - M2 + M3 + M6
        
        # For each Strassen term k (0 to 6), we add a rank-1 component.
        for k in range(7):
            col = base_rank + k
            
            # Define coefficients for A, B, C for this k-th Strassen product
            # A_coeff: 2x2 matrix of coefficients for A_block
            # B_coeff: 2x2 matrix of coefficients for B_block
            # C_coeff: 2x2 matrix of coefficients for C_block
            
            if k == 0: # M1
                A_c = [[1, 0], [0, 1]]
                B_c = [[1, 0], [0, 1]]
                C_c = [[1, 0], [0, 1]]
            elif k == 1: # M2
                A_c = [[0, 0], [1, 1]]
                B_c = [[1, 0], [0, 0]]
                C_c = [[0, 0], [1, -1]] # P10 gets +1, P11 gets -1
            elif k == 2: # M3
                A_c = [[1, 0], [0, 0]]
                B_c = [[0, 1], [0, -1]]
                C_c = [[0, 1], [0, 1]] # P01 gets +1, P11 gets +1
            elif k == 3: # M4
                A_c = [[0, 0], [0, 1]]
                B_c = [[-1, 0], [1, 0]]
                C_c = [[1, 0], [1, 0]] # P00 gets +1, P10 gets +1
            elif k == 4: # M5
                A_c = [[1, 1], [0, 0]]
                B_c = [[0, 0], [0, 1]]
                C_c = [[-1, 1], [0, 0]] # P00 gets -1, P01 gets +1
            elif k == 5: # M6
                A_c = [[-1, 0], [1, 0]]
                B_c = [[1, 1], [0, 0]]
                C_c = [[0, 0], [0, 1]] # P11 gets +1
            elif k == 6: # M7
                A_c = [[0, 1], [0, -1]]
                B_c = [[0, 1], [1, 1]]
                C_c = [[1, 0], [0, 0]] # P00 gets +1
            
            # Update U (A factors)
            # A_block local (lr, lc) maps to global (lr, a_col_start + lc)
            for lr in range(2):
                for lc in range(2):
                    coeff = A_c[lr][lc]
                    if coeff != 0:
                        global_row = lr
                        global_col = a_col_start + lc
                        u_idx = global_row * m + global_col
                        U[u_idx, col] += coeff
            
            # Update V (B factors)
            # B_block local (lr, lc) maps to global (b_row_start + lr, b_col_start + lc)
            for lr in range(2):
                for lc in range(2):
                    coeff = B_c[lr][lc]
                    if coeff != 0:
                        global_row = b_row_start + lr
                        global_col = b_col_start + lc
                        v_idx = global_row * p + global_col
                        V[v_idx, col] += coeff
            
            # Update W (C factors)
            # C_block local (lr, lc) maps to global (c_col_start + lc, lr)
            # W index is k*n + i, where k is col, i is row
            for lr in range(2):
                for lc in range(2):
                    coeff = C_c[lr][lc]
                    if coeff != 0:
                        global_row = lr
                        global_col = c_col_start + lc
                        w_idx = global_col * n + global_row
                        W[w_idx, col] += coeff

    # Block 1: A_left * B_top_left -> C_left_part1
    # A_left: cols 0,1. B_top_left: rows 0,1 cols 0,1. C_left: cols 0,1.
    # Base rank: 0
    apply_strassen_block(0, 0, 0, 0, 0)
    
    # Block 2: A_right * B_bottom_left -> C_left_part2
    # A_right: cols 2,3. B_bottom_left: rows 2,3 cols 0,1. C_left: cols 0,1.
    # Base rank: 7
    apply_strassen_block(7, 2, 2, 0, 0)
    
    # Block 3: A_left * B_top_right -> C_right_part1
    # A_left: cols 0,1. B_top_right: rows 0,1 cols 2,3. C_right: cols 2,3.
    # Base rank: 14
    apply_strassen_block(14, 0, 0, 2, 2)
    
    # Block 4: A_right * B_bottom_right -> C_right_part2
    # A_right: cols 2,3. B_bottom_right: rows 2,3 cols 2,3. C_right: cols 2,3.
    # Base rank: 21
    apply_strassen_block(21, 2, 2, 2, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = 2, 4, 4
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Verification
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf):
        print("Mismatch")
        print(C)
        print(Cf)
    else:
        print("OK rank", U.shape[1])


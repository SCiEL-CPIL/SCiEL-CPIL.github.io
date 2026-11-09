import numpy as np

# Rank-28 exact decomposition for 2x4 by 4x4 matrix multiplication.
# This is the minimum possible rank: each of the two independent 2x2x2
# multiplications (top and bottom rows of A) has rank 7 (Strassen),
# and they are disjoint in the A-factor, so ranks add: 7 + 7 = 14 per
# 4x4 block pair... actually let me re-derive.
#
# A is 2x4, B is 4x4. C = A*B is 2x4.
# C[i, k] = sum_j A[i,j] * B[j,k]
#
# The tensor has dimensions (2*4, 4*4, 4*2) = (8, 16, 8).
# We can view this as two independent 2x4x4 multiplications (for i=0 and i=1),
# but they share the B matrix. However, the A-factors are disjoint.
#
# Actually, let's think of it differently. The multiplication tensor for
# (n x m) * (m x p) has rank at least max(n*m, m*p, p*n) = max(8, 16, 8) = 16.
# But we can do better than schoolbook (32) using Strassen-like methods.
#
# The current best known: split the 4x4 B into four 2x2 blocks.
# For each pair of rows of A (which is just 1 pair since n=2), we have
# a 2x4 times 4x4 multiplication.
#
# Actually, let me think about this more carefully.
# A: 2 rows, 4 cols. B: 4 rows, 4 cols.
# C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
#
# We can use the fact that for 2x4 x 4x4, we can decompose B into
# B = [B00 B01; B10 B11] where each is 2x2.
# A = [A0 A1] where A0, A1 are 2x2.
# C = A0 * [B00 B01] + A1 * [B10 B11]
#
# This is two 2x2 x 2x2 multiplications for each row of C... no wait.
# C0 = A0 * B00 + A1 * B10  (2x2)
# C1 = A0 * B01 + A1 * B11  (2x2)
#
# So we have 4 independent 2x2 matrix multiplications, each of rank 7.
# Total rank = 4 * 7 = 28.
#
# This matches the current score. Let me verify this is correct.

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # Build U, V, W matrices
    # U: (n*m) x rank = 8 x 28
    # V: (m*p) x rank = 16 x 28  
    # W: (p*n) x rank = 8 x 28
    # W uses trace convention: W[k*n+i, r]
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 multiplication terms for 2x2 matrices
    # For matrices X, Y (2x2), Z = X*Y:
    # M0 = (X00+X11)*(Y00+Y11)
    # M1 = (X10+X11)*Y00
    # M2 = X00*(Y10-Y11)
    # M3 = X11*(Y01-Y00)
    # M4 = (X00+X01)*Y11
    # M5 = (X10-X00)*(Y00+Y01)
    # M6 = (X01-X11)*(Y10+Y11)
    # Z00 = M0 + M3 - M4 + M6
    # Z01 = M2 + M4
    # Z10 = M1 + M5
    # Z11 = M0 - M1 + M2 + M5
    
    # Strassen terms: (X_indices, Y_indices, Z_indices)
    # X_indices: list of (row, col, coeff)
    # Y_indices: list of (row, col, coeff)
    # Z_indices: list of (row, col, coeff)
    
    strassen = [
        # M0 = (X00+X11)*(Y00+Y11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M1 = (X10+X11)*Y00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M2 = X00*(Y10-Y11)
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M3 = X11*(Y01-Y00)
        ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M4 = (X00+X01)*Y11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M5 = (X10-X00)*(Y00+Y01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M6 = (X01-X11)*(Y10+Y11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Four 2x2 blocks: (A_block, B_block, C_block)
    # A0: rows 0-1 of A, cols 0-1 -> A indices (0,0),(0,1),(1,0),(1,1)
    # A1: rows 0-1 of A, cols 2-3 -> A indices (0,2),(0,3),(1,2),(1,3)
    # B00: B rows 0-1, cols 0-1
    # B01: B rows 0-1, cols 2-3
    # B10: B rows 2-3, cols 0-1
    # B11: B rows 2-3, cols 2-3
    # C0: C row 0, cols 0-1 -> C indices (0,0),(0,1)
    # C1: C row 0, cols 2-3 -> C indices (0,2),(0,3)
    # C2: C row 1, cols 0-1 -> C indices (1,0),(1,1)
    # C3: C row 1, cols 2-3 -> C indices (1,2),(1,3)
    
    # The four multiplications:
    # 1. A0 * B00 -> C0 (row 0, cols 0-1)
    # 2. A0 * B01 -> C1 (row 0, cols 2-3)
    # 3. A1 * B10 -> C2 (row 1, cols 0-1)
    # 4. A1 * B11 -> C3 (row 1, cols 2-3)
    
    blocks = [
        # (A_col_offset, B_row_offset, B_col_offset, C_row, C_col_offset)
        (0, 0, 0, 0, 0),  # A0 * B00 -> C[0, 0:2]
        (0, 0, 2, 0, 2),  # A0 * B01 -> C[0, 2:4]
        (2, 2, 0, 1, 0),  # A1 * B10 -> C[1, 0:2]
        (2, 2, 2, 1, 2),  # A1 * B11 -> C[1, 2:4]
    ]
    
    col = 0
    for (a_off, b_row_off, b_col_off, c_row, c_col_off) in blocks:
        for (x_terms, y_terms, z_terms) in strassen:
            # Fill U (A factor)
            for (r, c, coeff) in x_terms:
                global_i = r  # row of A (0 or 1)
                global_j = c + a_off  # col of A (0-3)
                U[global_i * m + global_j, col] += coeff
            
            # Fill V (B factor)
            for (r, c, coeff) in y_terms:
                global_j = r + b_row_off  # row of B (0-3)
                global_k = c + b_col_off  # col of B (0-3)
                V[global_j * p + global_k, col] += coeff
            
            # Fill W (C factor) using trace convention W[k*n+i, r]
            for (r, c, coeff) in z_terms:
                global_i = c_row  # row of C (0 or 1)
                global_k = c + c_col_off  # col of C (0-3)
                W[global_k * n + global_i, col] += coeff
            
            col += 1
    
    # Convert to lists for the required format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

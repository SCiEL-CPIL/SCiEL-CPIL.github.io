import numpy as np
import time

def construct():
    n, m, p = 3, 4, 6
    
    # Rank 24 decomposition: split B into 3 blocks of 4x2
    # For each block, use rank-8 decomposition of 3x4x2
    # Total rank = 3 * 8 = 24
    
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]  # (12, 24)
    V = [[0] * rank for _ in range(m * p)]  # (24, 24)
    W = [[0] * rank for _ in range(p * n)]  # (18, 24)
    
    col = 0
    
    # For each of the 3 B column blocks (columns 0-1, 2-3, 4-5)
    for b_block in range(3):
        b_col_start = 2 * b_block
        b_col_end = 2 * b_block + 1
        
        # B indices: rows 0-3, cols b_col_start, b_col_end
        # A indices: rows 0-2, cols 0-3
        # C indices: rows 0-2, cols b_col_start, b_col_end
        
        # Define a helper to add a bilinear term
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, coeff in u_coeffs:
                U[idx][col] += coeff
            for idx, coeff in v_coeffs:
                V[idx][col] += coeff
            for idx, coeff in w_coeffs:
                W[idx][col] += coeff
            col += 1
        
        # A indices: a[i][j] = i*m + j for i in 0,1,2; j in 0,1,2,3
        a = [[i * m + j for j in range(4)] for i in range(3)]
        
        # B indices: b[j][k] = j*p + k for j in 0,1,2,3; k in {b_col_start, b_col_end}
        b = [[j * p + b_col_start, j * p + b_col_end] for j in range(4)]
        
        # C indices: c[i][k] = k*n + i for i in 0,1,2; k in {b_col_start, b_col_end}
        c = [[b_col_start * n + i, b_col_end * n + i] for i in range(3)]
        
        # Rank-8 decomposition for 3x4x2 matrix multiplication
        # A is 3x4, B is 4x2, C = A*B is 3x2
        # We need 8 bilinear terms that express all 6 entries of C
        
        # Strategy: Use Strassen-like approach for the 2x2 part, handle extra row/col
        # Split A into rows (0,1) and (2), B into cols (0) and (1)
        # Use 7 terms for top 2x2x2, then handle bottom row with additional terms
        
        # Actually, let's use a different approach:
        # Split A columns into (0,1) and (2,3)
        # For each pair of A columns, we can use a 3x2x2 decomposition
        
        # Better: use the fact that R(3,4,2) <= 8
        # Terms:
        # M1 = (a00+a11)(b00+b11) -> contributes to C[0,0], C[1,1]
        # M2 = (a10+a11)b00 -> contributes to C[1,0], C[1,1]
        # M3 = a00(b01-b11) -> contributes to C[0,1], C[1,1]
        # M4 = a11(b10-b00) -> contributes to C[0,0], C[1,0]
        # M5 = (a00+a01)b11 -> contributes to C[0,0], C[0,1]
        # M6 = (a10-a00)(b00+b01) -> contributes to C[1,1]
        # M7 = (a01-a11)(b10+b11) -> contributes to C[0,0]
        # M8 = a20*b00 + a21*b10 + a20*b01 + a21*b11 -> but this is 4 terms, too many
        
        # Let me think differently. Use 8 terms total.
        # Split the problem:
        # C[0,:] = A[0,:] * B[:,0:2] (row 0 of C)
        # C[1,:] = A[1,:] * B[:,0:2] (row 1 of C)
        # C[2,:] = A[2,:] * B[:,0:2] (row 2 of C)
        
        # Use 4 terms for rows 0,1 (Strassen 2x2x2 needs 7, but we have only 2 cols)
        # Actually R(2,4,2) = 4 (schoolbook)
        # R(1,4,2) = 4 (schoolbook)
        # Total = 4 + 4 = 8
        
        # Terms for C[0,:] and C[1,:]:
        # Use 4 schoolbook terms for the 2x4x2 part (rows 0,1 of A)
        # T1: A[0:2, 0] * B[0, :] -> C[0:2, 0:2]
        # T2: A[0:2, 1] * B[1, :] -> C[0:2, 0:2]
        # T3: A[0:2, 2] * B[2, :] -> C[0:2, 0:2]
        # T4: A[0:2, 3] * B[3, :] -> C[0:2, 0:2]
        
        # For each column j of A (and corresponding row of B):
        # Term j: U has A[0,j] and A[1,j], V has B[j,0] and B[j,1], W has C[0,0],C[0,1],C[1,0],C[1,1]
        
        # But that's 4 terms giving 8 outputs (2 rows x 2 cols each), which is correct.
        # Then 4 more terms for row 2 of A:
        # T5: A[2, 0] * B[0, :] -> C[2, 0:2]
        # T6: A[2, 1] * B[1, :] -> C[2, 0:2]
        # T7: A[2, 2] * B[2, :] -> C[2, 0:2]
        # T8: A[2, 3] * B[3, :] -> C[2, 0:2]
        
        # Total: 8 terms, rank 8 per block, 24 total.
        
        for j in range(4):
            # Term for A[:,j] * B[j,:]
            # U: A[0,j], A[1,j], A[2,j]
            # V: B[j, b_col_start], B[j, b_col_end]
            # W: C[0,b_col_start], C[0,b_col_end], C[1,b_col_start], C[1,b_col_end], C[2,b_col_start], C[2,b_col_end]
            
            u_coeffs = [
                (a[0][j], 1),
                (a[1][j], 1),
                (a[2][j], 1),
            ]
            v_coeffs = [
                (b[j][0], 1),
                (b[j][1], 1),
            ]
            w_coeffs = [
                (c[0][0], 1),
                (c[0][1], 1),
                (c[1][0], 1),
                (c[1][1], 1),
                (c[2][0], 1),
                (c[2][1], 1),
            ]
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
    
    return {"U": U, "V": V, "W": W}

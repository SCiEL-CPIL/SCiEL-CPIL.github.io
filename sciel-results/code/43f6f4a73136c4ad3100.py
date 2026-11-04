"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 8 rows
    V = [[0] * rank for _ in range(m * p)] # 16 rows
    W = [[0] * rank for _ in range(p * n)] # 8 rows

    # Winograd coefficients for 2x2x2
    # U coefficients (rows of A, 0-1)
    # V coefficients (rows of B, 0-1)
    # W coefficients (cols of C, 0-1)
    # Note: W index in task is k*n + i. For 2x2 block, k is col index (0-1), i is row index (0-1).
    
    # M1: (a00+a11)(b00+b11)
    # M2: (a01+a11)(b10-b00)
    # M3: (a00-a10)(b00+b10)
    # M4: a11(b11-b00)
    # M5: (a00+a01)(b11-b01)
    # M6: (a10+a11)(b01-b11)
    # M7: (a00-a01+a10)(b00-b01+b11)
    
    # C00 = M1 + M2 + M3 - M4 - M5
    # C01 = M2 + M4 + M5 - M6 - M7
    # C10 = M1 - M2 - M3 + M6 + M7
    # C11 = M1 + M3 - M4 + M5 - M7

    u_coeffs = [
        [1, 0, 1, 0, 1, 0, 1], # a00
        [0, 1, 0, -1, 1, 1, -1], # a10? No, indices:
        # Let's map A indices:
        # a00 is A[0,0]
        # a10 is A[1,0]
        # a01 is A[0,1]
        # a11 is A[1,1]
        
        # M1: a00=1, a11=1
        # M2: a01=1, a11=1
        # M3: a00=1, a10=-1
        # M4: a11=1
        # M5: a00=1, a01=1
        # M6: a10=1, a11=1
        # M7: a00=1, a01=-1, a10=1
        
        # Row 0 (a00): 1, 0, 1, 0, 1, 0, 1
        # Row 1 (a10): 0, 0, -1, 0, 0, 1, 1
        # Row 2 (a01): 0, 1, 0, 0, 1, 0, -1
        # Row 3 (a11): 1, 1, 0, 1, 0, 1, 0
    ]
    
    v_coeffs = [
        # b00, b10, b01, b11
        # M1: b00=1, b11=1
        # M2: b10=1, b00=-1
        # M3: b00=1, b10=1
        # M4: b11=1, b00=-1
        # M5: b11=1, b01=-1
        # M6: b01=1, b11=-1
        # M7: b00=1, b01=-1, b11=1
        
        # Row 0 (b00): 1, -1, 1, -1, 0, 0, 1
        # Row 1 (b10): 0, 1, 1, 0, 0, 0, 0
        # Row 2 (b01): 0, 0, 0, 0, -1, 1, -1
        # Row 3 (b11): 1, 0, 0, 1, 1, -1, 1
    ]
    
    w_coeffs = [
        # c00, c01, c10, c11
        # M1: 1, 0, 1, 1
        # M2: 1, 1, -1, 0
        # M3: 1, 0, -1, 1
        # M4: -1, 1, 0, -1
        # M5: -1, 1, 0, 1
        # M6: 0, -1, 1, 0
        # M7: 0, -1, 1, -1
        
        # Col 0 (c00): 1, 1, 1, -1, -1, 0, 0
        # Col 1 (c01): 0, 1, 0, 1, 1, -1, -1
        # Col 2 (c10): 1, -1, -1, 0, 0, 1, 1
        # Col 3 (c11): 1, 0, 1, -1, 1, 0, -1
    ]
    
    # Blocks:
    # 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 1: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    block_configs = [
        # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
        # A is 2x4. Rows 0-1. Cols 0-3.
        # B is 4x4. Rows 0-3. Cols 0-3.
        # C is 2x4. Rows 0-1. Cols 0-3.
        # W index is k*n + i. k is col, i is row.
        
        # Block 0
        (0, 0, 0, 0, 0, 0),
        # Block 1
        (0, 0, 0, 2, 0, 2),
        # Block 2
        (0, 2, 2, 0, 0, 0),
        # Block 3
        (0, 2, 2, 2, 0, 2),
    ]
    
    col = 0
    for config in block_configs:
        a_r_off, a_c_off, b_r_off, b_c_off, c_r_off, c_c_off = config
        
        for m_idx in range(7):
            # U
            for a_r in range(2):
                coeff = u_coeffs[a_r][m_idx]
                if coeff != 0:
                    u_row = (a_r + a_r_off) * m + (a_c_off + 0) # Wait, a_c_off is 0 or 2.
                    # A index: row i, col j.
                    # In block, local col is 0 or 1. Global col is a_c_off + local_col.
                    # u_coeffs row index corresponds to local row?
                    # u_coeffs[0] is a00 (row 0, col 0).
                    # u_coeffs[1] is a10 (row 1, col 0).
                    # u_coeffs[2] is a01 (row 0, col 1).
                    # u_coeffs[3] is a11 (row 1, col 1).
                    
                    # So local row = a_r, local col = a_c
                    # u_coeffs index mapping:
                    # 0: r0c0
                    # 1: r1c0
                    # 2: r0c1
                    # 3: r1c1
                    
                    # My u_coeffs list above:
                    # Row 0: a00
                    # Row 1: a10
                    # Row 2: a01
                    # Row 3: a11
                    
                    # So for local row r, local col c:
                    # if r=0, c=0 -> idx 0
                    # if r=1, c=0 -> idx 1
                    # if r=0, c=1 -> idx 2
                    # if r=1, c=1 -> idx 3
                    
                    # Let's re-verify u_coeffs structure.
                    # I defined u_coeffs as list of rows.
                    # u_coeffs[0] corresponds to local row 0? No, I listed a00, a10, a01, a11.
                    # This is not row-major. It's mixed.
                    # Let's make u_coeffs indexed by [local_row][local_col].
                    
            # Let's restart the coefficient definition with proper [row][col] structure.
            pass
            
    # Redefine coefficients properly
    # U: 2x2
    # u[0][0] (a00): 1, 0, 1, 0, 1, 0, 1
    # u[1][0] (a10): 0, 0, -1, 0, 0, 1, 1
    # u[0][1] (a01): 0, 1, 0, 0, 1, 0, -1
    # u[1][1] (a11): 1, 1, 0, 1, 0, 1, 0
    u_mat = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, -1, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, -1],
        [1, 1, 0, 1, 0, 1, 0]
    ]
    # Flatten to [row][col]
    u_local = [
        [u_mat[0], u_mat[2]],
        [u_mat[1], u_mat[3]]
    ]
    
    # V: 2x2
    # v[0][0] (b00): 1, -1, 1, -1, 0, 0, 1
    # v[1][0] (b10): 0, 1, 1, 0, 0, 0, 0
    # v[0][1] (b01): 0, 0, 0, 0, -1, 1, -1
    # v[1][1] (b11): 1, 0, 0, 1, 1, -1, 1
    v_mat = [
        [1, -1, 1, -1, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, -1],
        [1, 0, 0, 1, 1, -1, 1]
    ]
    v_local = [
        [v_mat[0], v_mat[2]],
        [v_mat[1], v_mat[3]]
    ]
    
    # W: 2x2 (Result C is 2x2 local)
    # w[0][0] (c00): 1, 1, 1, -1, -1, 0, 0
    # w[1][0] (c10): 1, -1, -1, 0, 0, 1, 1
    # w[0][1] (c01): 0, 1, 0, 1, 1, -1, -1
    # w[1][1] (c11): 1, 0, 1, -1, 1, 0, -1
    w_mat = [
        [1, 1, 1, -1, -1, 0, 0],
        [1, -1, -1, 0, 0, 1, 1],
        [0, 1, 0, 1, 1, -1, -1],
        [1, 0, 1, -1, 1, 0, -1]
    ]
    w_local = [
        [w_mat[0], w_mat[2]],
        [w_mat[1], w_mat[3]]
    ]
    
    col = 0
    for config in block_configs:
        a_r_off, a_c_off, b_r_off, b_c_off, c_r_off, c_c_off = config
        
        for m_idx in range(7):
            # U
            for a_r in range(2):
                for a_c in range(2):
                    coeff = u_local[a_r][a_c][m_idx]
                    if coeff != 0:
                        u_row = (a_r + a_r_off) * m + (a_c_off + a_c)
                        U[u_row][col] += coeff
            
            # V
            for b_r in range(2):
                for b_c in range(2):
                    coeff = v_local[b_r][b_c][m_idx]
                    if coeff != 0:
                        v_row = (b_r + b_r_off) * p + (b_c_off + b_c)
                        V[v_row][col] += coeff
                        
            # W
            for c_r in range(2):
                for c_c in range(2):
                    coeff = w_local[c_r][c_c][m_idx]
                    if coeff != 0:
                        # W index is k*n + i. k is col, i is row.
                        # Local c_r is row i, local c_c is col k.
                        w_row = (c_c + c_c_off) * n + (c_r + c_r_off)
                        W[w_row][col] += coeff
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

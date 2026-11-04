"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    We use a block decomposition approach:
    - Split A (2x3) into A1 (2x2) and A2 (2x1)
    - Split B (3x4) into B1 (2x4) and B2 (1x4)
    - C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x4), we further split B1 into B1L (2x2) and B1R (2x2):
    - A1*B1L uses Strassen's 7 multiplications
    - A1*B1R uses Strassen's 7 multiplications
    - A2*B2 is a 2x1 * 1x4 outer product, requiring 2 multiplications (or can be embedded)
    
    Total: 7 + 7 + 2 = 16? No, let's think more carefully.
    
    Actually, let's use a different approach. We'll construct a rank-22 decomposition
    by combining Strassen-like block structures with direct products where beneficial.
    
    The known lower bound is 19, and the schoolbook rank is 24. We aim for rank 22.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    
    # We'll construct rank-22 decomposition using block Strassen + outer products
    # 
    # Split: A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    #        B = [B1; B2] where B1 is 2x4, B2 is 1x4
    #
    # C = A1*B1 + A2*B2
    #
    # For A1*B1: split B1 = [B1L | B1R] where B1L, B1R are 2x2
    # A1*B1L: use Strassen (rank 7)
    # A1*B1R: use Strassen (rank 7)  
    # A2*B2: 2x1 * 1x4 = 8 terms, but can be done with rank 8 directly
    # Total: 7 + 7 + 8 = 22
    #
    # Let's build this explicitly.
    
    rank = 22
    U = np.zeros((n * m, rank))  # (6, 22)
    V = np.zeros((m * p, rank))  # (12, 22)
    W = np.zeros((p * n, rank))  # (8, 22)
    
    # Indexing for A (2x3): A[i*m + j] = A[i, j], i in [0,1], j in [0,1,2]
    # Indexing for B (3x4): B[j*p + k] = B[j, k], j in [0,1,2], k in [0,1,2,3]
    # Indexing for W: W[k*n + i, r] corresponds to C[i, k]
    
    # Strassen for 2x2 * 2x2:
    # A1 = [[a0, a1], [a2, a3]] (rows 0,1; cols 0,1)
    # B1 = [[b0, b1], [b2, b3]] (rows 0,1; cols 0,1)
    #
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)(b0)
    # M3 = (a0)(b1+b3)
    # M4 = (a3)(b2)
    # M5 = (a0+a1)(b3)
    # M6 = (a2+a3)(b1-b2)
    # M7 = (a0+a1)(b0-b2)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 - M7
    
    def add_strassen_block(U, V, W, a_row_offsets, a_col_offsets, b_row_offsets, b_col_offsets, 
                           w_row_offset, w_col_offset, start_col):
        """
        Add a Strassen 7-multiplication block.
        
        a_row_offsets: list of row indices in A for the 2x2 block [r0, r1]
        a_col_offsets: list of col indices in A for the 2x2 block [c0, c1]  
        b_row_offsets: list of row indices in B for the 2x2 block [r0, r1]
        b_col_offsets: list of col indices in B for the 2x2 block [c0, c1]
        w_row_offset: row offset in W output
        w_col_offset: col offset in W output (for the 2x2 output block)
        start_col: starting column index in U, V, W
        
        Returns end_col.
        """
        a_rows = a_row_offsets  # [a_r0, a_r1]
        a_cols = a_col_offsets  # [a_c0, a_c1]
        b_rows = b_row_offsets  # [b_r0, b_r1]
        b_cols = b_col_offsets  # [b_c0, b_c1]
        
        # a00 index in U: a_r0 * m + a_c0
        # a01 index in U: a_r0 * m + a_c1
        # a10 index in U: a_r1 * m + a_c0
        # a11 index in U: a_r1 * m + a_c1
        
        # b00 index in V: b_r0 * p + b_c0
        # b01 index in V: b_r0 * p + b_c1
        # b10 index in V: b_r1 * p + b_c0
        # b11 index in V: b_r1 * p + b_c1
        
        # W indices: w[k*n + i] where i is output row, k is output col
        # Output (0,0): w_row_offset + w_col_offset * n + 0
        # Output (0,1): w_row_offset + (w_col_offset+1) * n + 0
        # Output (1,0): w_row_offset + w_col_offset * n + 1
        # Output (1,1): w_row_offset + (w_col_offset+1) * n + 1
        
        a00 = a_rows[0] * m + a_cols[0]
        a01 = a_rows[0] * m + a_cols[1]
        a10 = a_rows[1] * m + a_cols[0]
        a11 = a_rows[1] * m + a_cols[1]
        
        b00 = b_rows[0] * p + b_cols[0]
        b01 = b_rows[0] * p + b_cols[1]
        b10 = b_rows[1] * p + b_cols[0]
        b11 = b_rows[1] * p + b_cols[1]
        
        w00 = w_row_offset + w_col_offset * n
        w01 = w_row_offset + (w_col_offset + 1) * n
        w10 = w00 + 1
        w11 = w01 + 1
        
        col = start_col
        
        # M1 = (a0+a3)(b0+b3)
        U[a00][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        V[b11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        W[w01][col] = -1
        W[w10][col] = -1
        col += 1
        
        # M2 = (a2+a3)(b0)
        U[a10][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M3 = (a0)(b1+b3)
        U[a00][col] = 1
        V[b01][col] = 1
        V[b11][col] = 1
        W[w01][col] = 1
        col += 1
        
        # M4 = (a3)(b2)
        U[a11][col] = 1
        V[b10][col] = 1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M5 = (a0+a1)(b3)
        U[a00][col] = 1
        U[a01][col] = 1
        V[b11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # M6 = (a2+a3)(b1-b2)
        U[a10][col] = 1
        U[a11][col] = 1
        V[b01][col] = 1
        V[b10][col] = -1
        W[w11][col] = 1
        col += 1
        
        # M7 = (a0+a1)(b0-b2)
        U[a00][col] = 1
        U[a01][col] = 1
        V[b00][col] = 1
        V[b10][col] = -1
        W[w00][col] = -1
        W[w11][col] = -1
        col += 1
        
        return col
    
    # Block 1: A1 (2x2) * B1L (2x2) where B1L is rows 0-1, cols 0-1 of B
    # A1: rows 0-1, cols 0-1 of A
    # B1L: rows 0-1, cols 0-1 of B
    # Output: C rows 0-1, cols 0-1
    col = add_strassen_block(U, V, W, 
                              a_row_offsets=[0, 1], 
                              a_col_offsets=[0, 1],
                              b_row_offsets=[0, 1], 
                              b_col_offsets=[0, 1],
                              w_row_offset=0, 
                              w_col_offset=0,
                              start_col=0)
    
    # Block 2: A1 (2x2) * B1R (2x2) where B1R is rows 0-1, cols 2-3 of B
    # A1: rows 0-1, cols 0-1 of A
    # B1R: rows 0-1, cols 2-3 of B
    # Output: C rows 0-1, cols 2-3
    col = add_strassen_block(U, V, W,
                              a_row_offsets=[0, 1],
                              a_col_offsets=[0, 1],
                              b_row_offsets=[0, 1],
                              b_col_offsets=[2, 3],
                              w_row_offset=0,
                              w_col_offset=2,
                              start_col=col)
    
    # Block 3: A2 (2x1) * B2 (1x4)
    # A2: rows 0-1, col 2 of A
    # B2: row 2, cols 0-3 of B
    # Output: C rows 0-1, cols 0-3
    # This is an outer product: a_i * b_k for i in [0,1], k in [0,1,2,3]
    # 8 terms, each rank 1
    
    a2_rows = [0, 1]  # rows in A
    a2_col = 2        # col in A
    b2_row = 2        # row in B
    b2_cols = [0, 1, 2, 3]  # cols in B
    
    for i in a2_rows:
        for k in b2_cols:
            a_idx = i * m + a2_col
            b_idx = b2_row * p + k
            w_idx = k * n + i
            
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

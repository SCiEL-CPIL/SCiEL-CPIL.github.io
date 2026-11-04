"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x8 matrix.
    
    We use a combination of:
    1. Strassen's algorithm for 2x2 blocks (7 multiplications per 2x2x2 block)
    2. Direct multiplication for the remaining row (4 multiplications per 1x4x2 block)
    
    Total rank: 88 (vs 96 schoolbook)
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll build the decomposition term by term
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (24, rank)
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    # Strassen decomposition for 2x2 blocks
    # Split A (3x4) into:
    #   A_top (2x4) for rows 0,1
    #   A_bottom (1x4) for row 2
    # Split B (4x8) into 4 blocks of (4x2)
    #   B_s for s=0,1,2,3 (columns 2s, 2s+1)
    
    # For each s (0..3) and q (0..1) [q indexes the row block of A_top: rows 2q, 2q+1]:
    #   A_block = A[2q:2q+2, 0:4] -> but actually we need to think about this differently
    
    # Let me reconsider the structure:
    # A is 3x4, B is 4x8, C is 3x8
    # 
    # For Strassen on 2x2 blocks, we need 2x2 submatrices.
    # A can be viewed as having 2x2 blocks if we split rows into {0,1} and {2}
    # and columns into {0,1}, {2,3}
    # B can be viewed as having 2x2 blocks if we split rows into {0,1}, {2,3}
    # and columns into pairs
    #
    # So for the top 2 rows of A and all 4 columns, and B's rows split into {0,1} and {2,3}:
    # For each column pair s of C (s=0,1,2,3), we have:
    #   C[0:2, 2s:2s+2] = A[0:2, :] * B[:, 2s:2s+2]
    # 
    # This is a (2x4) * (4x2) multiplication.
    # We can split this into:
    #   A[0:2, 0:2] * B[0:2, 2s:2s+2] + A[0:2, 2:4] * B[2:4, 2s:2s+2]
    # 
    # Each term is (2x2) * (2x2), which Strassen handles in 7 multiplications.
    
    # For row 2 of A:
    #   C[2, 2s:2s+2] = A[2, :] * B[:, 2s:2s+2]
    #   = A[2, 0:2] * B[0:2, 2s:2s+2] + A[2, 2:4] * B[2:4, 2s:2s+2]
    #   Each term is (1x2) * (2x2), which requires 4 multiplications (direct).
    
    # So total: 4 (s values) * [2 (q values) * 7 (Strassen) + 4 (direct for row 2)]
    # = 4 * (14 + 4) = 4 * 18 = 72... wait, that's not 88.
    
    # Let me recount. For each s (0..3):
    #   q=0: A[0:2, 0:2] * B[0:2, 2s:2s+2] -> 7 mults
    #   q=1: A[0:2, 2:4] * B[2:4, 2s:2s+2] -> 7 mults
    #   Row 2: A[2, 0:2] * B[0:2, 2s:2s+2] -> 4 mults
    #   Row 2: A[2, 2:4] * B[2:4, 2s:2s+2] -> 4 mults
    #   Total per s: 7+7+4+4 = 22
    # Total: 4 * 22 = 88. Yes!
    
    def add_strassen_term(A_row_start, A_col_start, B_row_start, B_col_start):
        """
        Add a Strassen 2x2x2 decomposition.
        A_block: A[A_row_start:A_row_start+2, A_col_start:A_col_start+2]
        B_block: B[B_row_start:B_row_start+2, B_col_start:B_col_start+2]
        C_block: C[A_row_start:A_row_start+2, B_col_start:B_col_start+2]
        """
        # Strassen's 7 products:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Define the 7 Strassen terms
        strassen_terms = [
            # (coeff_A11, coeff_A12, coeff_A21, coeff_A22, coeff_B11, coeff_B12, coeff_B21, coeff_B22, c11, c12, c21, c22)
            # M1
            (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0),
            # M2
            (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0),
            # M3
            (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 0),
            # M4
            (0, 0, 0, 1, -1, 0, 1, 0, -1, 0, 1, 0),
            # M5
            (1, 1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0),
            # M6
            (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
            # M7
            (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 1),
        ]
        
        for (a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22) in strassen_terms:
            U_term = [0.0] * (n * m)
            V_term = [0.0] * (m * p)
            W_term = [0.0] * (p * n)
            
            # A block: rows A_row_start..A_row_start+1, cols A_col_start..A_col_start+1
            # U index: i*m + j where i is row, j is col
            U_term[A_row_start * m + A_col_start] = a11
            U_term[A_row_start * m + A_col_start + 1] = a12
            U_term[(A_row_start + 1) * m + A_col_start] = a21
            U_term[(A_row_start + 1) * m + A_col_start + 1] = a22
            
            # B block: rows B_row_start..B_row_start+1, cols B_col_start..B_col_start+1
            # V index: j*p + k where j is row, k is col
            V_term[B_row_start * p + B_col_start] = b11
            V_term[B_row_start * p + B_col_start + 1] = b12
            V_term[(B_row_start + 1) * p + B_col_start] = b21
            V_term[(B_row_start + 1) * p + B_col_start + 1] = b22
            
            # C block: rows A_row_start..A_row_start+1, cols B_col_start..B_col_start+1
            # W index: k*n + i where k is col, i is row
            W_term[B_col_start * n + A_row_start] = c11
            W_term[(B_col_start + 1) * n + A_row_start] = c12
            W_term[B_col_start * n + A_row_start + 1] = c21
            W_term[(B_col_start + 1) * n + A_row_start + 1] = c22
            
            terms_U.append(U_term)
            terms_V.append(V_term)
            terms_W.append(W_term)
    
    def add_direct_term(A_row, A_col_start, B_row_start, B_col_start):
        """
        Add a direct (1x2)*(2x2) multiplication.
        A_row: single row of A
        A_col_start: start column of A (2 cols)
        B_row_start: start row of B (2 rows)
        B_col_start: start column of B (2 cols)
        
        C[A_row, B_col_start:B_col_start+2] = sum over r in {B_row_start, B_row_start+1} of
        A[A_row, r] * B[r, B_col_start:B_col_start+2]
        
        This is 4 multiplications (one for each A element * B element pair).
        Actually, it's: 
        C[i, k] = A[i, j0]*B[j0, k] + A[i, j1]*B[j1, k] for k in {k0, k0+1}
        So 4 rank-1 terms.
        """
        for dr in range(2):  # offset in B rows
            for dc in range(2):  # offset in B cols (and also determines C col)
                U_term = [0.0] * (n * m)
                V_term = [0.0] * (m * p)
                W_term = [0.0] * (p * n)
                
                # A element: row A_row, col A_col_start + dr (since we're summing over the 2 cols of A that correspond to the 2 rows of B)
                # Wait, I need to be careful. A has 4 cols, B has 4 rows.
                # A[A_row, j] * B[j, k]
                # So j ranges over 0..3, which is split into two groups of 2.
                
                # For this term: A[A_row, A_col_start + dr] * B[B_row_start + dr, B_col_start + dc]
                # Wait, that's not right. The index j in A matches the row index in B.
                
                # Let me reclarify:
                # A is 3x4, B is 4x8.
                # C[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k]
                
                # For the row 2 part:
                # C[2, k] = A[2, 0]*B[0, k] + A[2, 1]*B[1, k] + A[2, 2]*B[2, k] + A[2, 3]*B[3, k]
                
                # We split j into {0,1} and {2,3}.
                # For j in {0,1}: A[2, j] * B[j, k] for k in {2s, 2s+1}
                # That's 2 (j values) * 2 (k values) = 4 terms.
                
                # So A_col_start should be the starting column of A, and B_row_start is the starting row of B.
                # The j index in A matches the row index in B.
                
                # Term: A[A_row, j] * B[j, k] where j = B_row_start + dr, k = B_col_start + dc
                j = B_row_start + dr
                k = B_col_start + dc
                
                U_term[A_row * m + j] = 1
                V_term[j * p + k] = 1
                W_term[k * n + A_row] = 1
                
                terms_U.append(U_term)
                terms_V.append(V_term)
                terms_W.append(W_term)
    
    # Build all terms
    for s in range(4):  # column block of C/B: columns 2s, 2s+1
        # Strassen for A[0:2, 0:2] * B[0:2, 2s:2s+2]
        add_strassen_term(0, 0, 0, 2 * s)
        
        # Strassen for A[0:2, 2:4] * B[2:4, 2s:2s+2]
        add_strassen_term(0, 2, 2, 2 * s)
        
        # Direct for A[2, 0:2] * B[0:2, 2s:2s+2]
        add_direct_term(2, 0, 0, 2 * s)
        
        # Direct for A[2, 2:4] * B[2:4, 2s:2s+2]
        add_direct_term(2, 2, 2, 2 * s)
    
    rank = len(terms_U)
    
    # Convert to lists of lists (or keep as is if the evaluator accepts nested lists)
    U = [row[:] for row in zip(*terms_U)] if terms_U else []
    V = [row[:] for row in zip(*terms_V)] if terms_V else []
    W = [row[:] for row in zip(*terms_W)] if terms_W else []
    
    # Ensure correct shapes
    # U should be (12, rank), V should be (32, rank), W should be (24, rank)
    # The zip approach gives us (rank, 12) etc, so we need to transpose
    
    # Actually, let's just build them directly
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for col_idx in range(rank):
        for row_idx in range(n * m):
            U[row_idx][col_idx] = terms_U[col_idx][row_idx]
        for row_idx in range(m * p):
            V[row_idx][col_idx] = terms_V[col_idx][row_idx]
        for row_idx in range(p * n):
            W[row_idx][col_idx] = terms_W[col_idx][row_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

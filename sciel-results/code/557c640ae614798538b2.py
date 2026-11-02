"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    U_dim = n * m  # 12
    V_dim = m * p  # 24
    W_dim = p * n  # 18
    
    # Build target tensor for verification
    target = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m + j, j*p + k, k*n + i] = 1.0
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the target tensor."""
        M = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
        for r in range(U.shape[1]):
            M += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(M, target, atol=1e-9)
    
    def to_integer_list(M):
        """Convert numpy array to list of lists with integer values."""
        return [[int(round(x)) for x in row] for row in M]
    
    # Build Strassen-based decomposition for 3x4 * 4x6
    # Strategy: split m=4 into 2+2, then each 3x2*2x6 subproblem uses
    # Strassen on 2x2*2x3 sub-blocks.
    
    # For 2x2 * 2x2 Strassen (7 terms):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x3: split B into B_left (2x1) and B_right (2x2)
    # A*B_left: 2x2 * 2x1, schoolbook rank 4
    # A*B_right: 2x2 * 2x2, Strassen rank 7
    # Total: 11
    
    # For 3x2 * 2x6: split A into A_top (1x2) and A_bot (2x2), 
    # split B into B_left (2x3) and B_right (2x3)
    # A_top * B_left: 1x2 * 2x3, rank 6
    # A_top * B_right: 1x2 * 2x3, rank 6
    # A_bot * B_left: 2x2 * 2x3, rank 11
    # A_bot * B_right: 2x2 * 2x3, rank 11
    # Total: 34
    
    # For 3x4 * 4x6: split into two 3x2 * 2x6 subproblems
    # Total: 34 + 34 = 68
    
    # Build the decomposition by collecting all rank-1 terms
    # Each term is (u_idx, v_idx, w_idx, coeff)
    
    terms = []  # list of (u_pos, v_pos, w_pos, coeff)
    
    def add_term(u_pos, v_pos, w_pos, coeff):
        terms.append((u_pos, v_pos, w_pos, coeff))
    
    # Helper for 1x2 * 2x3 schoolbook (rank 6)
    # A is 1x2, B is 2x3, C is 1x3
    # C[j] = sum_k A[0,k]*B[k,j]
    # In global coords: A indices are a_row, a_col; B indices are b_row, b_col; C index is c_col
    # u_pos = a_global, v_pos = b_global, w_pos = c_global
    def add_1x2x2x3(a_row, a_cols, b_rows, b_cols, c_col):
        """Add schoolbook terms for 1x2 * 2x3.
        a_row: global row index of the 1x1 A block
        a_cols: list of 2 global column indices for A
        b_rows: list of 2 global row indices for B
        b_cols: list of 3 global column indices for B
        c_col: global column index for C (single row)
        """
        for k in range(2):
            for j in range(3):
                add_term(a_row * m + a_cols[k], b_rows[k] * p + b_cols[j], c_col * n + 0, 1)
    
    # Helper for 2x2 * 2x1 schoolbook (rank 4)
    def add_2x2x2x1(a_rows, a_cols, b_row, b_col, c_row, c_col):
        """Add schoolbook terms for 2x2 * 2x1.
        a_rows: 2 global row indices for A
        a_cols: 2 global column indices for A
        b_row: global row index for B (single column)
        b_col: global column index for B
        c_row: 2 global row indices for C
        c_col: global column index for C
        """
        for i in range(2):
            for k in range(2):
                add_term(a_rows[i] * m + a_cols[k], b_row * p + b_col, c_row[i] * n + c_col, 1)
    
    # Helper for 2x2 * 2x2 Strassen (rank 7)
    def add_2x2x2x2_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add Strassen terms for 2x2 * 2x2.
        a_rows: 2 global row indices for A
        a_cols: 2 global column indices for A
        b_rows: 2 global row indices for B
        b_cols: 2 global column indices for B
        c_rows: 2 global row indices for C
        c_cols: 2 global column indices for C
        """
        # A11 = A[a_rows[0], a_cols[0]], A12 = A[a_rows[0], a_cols[1]]
        # A21 = A[a_rows[1], a_cols[0]], A22 = A[a_rows[1], a_cols[1]]
        # B11 = B[b_rows[0], b_cols[0]], B12 = B[b_rows[0], b_cols[1]]
        # B21 = B[b_rows[1], b_cols[0]], B22 = B[b_rows[1], b_cols[1]]
        
        A11 = (a_rows[0] * m + a_cols[0])
        A12 = (a_rows[0] * m + a_cols[1])
        A21 = (a_rows[1] * m + a_cols[0])
        A22 = (a_rows[1] * m + a_cols[1])
        
        B11 = (b_rows[0] * p + b_cols[0])
        B12 = (b_rows[0] * p + b_cols[1])
        B21 = (b_rows[1] * p + b_cols[0])
        B22 = (b_rows[1] * p + b_cols[1])
        
        C11 = (c_rows[0] * n + c_cols[0])
        C12 = (c_rows[0] * n + c_cols[1])
        C21 = (c_rows[1] * n + c_cols[0])
        C22 = (c_rows[1] * n + c_cols[1])
        
        # M1 = (A11+A22)*(B11+B22)
        add_term(A11, B11, C11, 1)
        add_term(A11, B22, C11, 1)
        add_term(A22, B11, C11, 1)
        add_term(A22, B22, C11, 1)
        add_term(A11, B11, C12, 1)
        add_term(A11, B22, C12, 1)
        add_term(A22, B11, C12, 1)
        add_term(A22, B22, C12, 1)
        add_term(A11, B11, C21, -1)
        add_term(A11, B22, C21, -1)
        add_term(A22, B11, C21, -1)
        add_term(A22, B22, C21, -1)
        add_term(A11, B11, C22, 1)
        add_term(A11, B22, C22, 1)
        add_term(A22, B11, C22, 1)
        add_term(A22, B22, C22, 1)
        
        # M2 = (A21+A22)*B11
        add_term(A21, B11, C21, 1)
        add_term(A22, B11, C21, 1)
        add_term(A21, B11, C22, -1)
        add_term(A22, B11, C22, -1)
        
        # M3 = A11*(B12-B22)
        add_term(A11, B12, C12, 1)
        add_term(A11, B22, C12, -1)
        add_term(A11, B12, C22, 1)
        add_term(A11, B22, C22, -1)
        
        # M4 = A22*(B21-B11)
        add_term(A22, B21, C11, 1)
        add_term(A22, B11, C11, -1)
        add_term(A22, B21, C21, 1)
        add_term(A22, B11, C21, -1)
        
        # M5 = (A11+A12)*B22
        add_term(A11, B22, C12, 1)
        add_term(A12, B22, C12, 1)
        add_term(A11, B22, C11, -1)
        add_term(A12, B22, C11, -1)
        
        # M6 = (A21-A11)*(B11+B12)
        add_term(A21, B11, C22, 1)
        add_term(A21, B12, C22, 1)
        add_term(A11, B11, C22, -1)
        add_term(A11, B12, C22, -1)
        
        # M7 = (A12-A22)*(B21+B22)
        add_term(A12, B21, C11, 1)
        add_term(A12, B22, C11, 1)
        add_term(A22, B21, C11, -1)
        add_term(A22, B22, C11, -1)
    
    # Now build the full 3x4 * 4x6 decomposition
    # Split m=4 into 2+2: A = [A_left (3x2), A_right (3x2)], B = [B_top (2x6); B_bottom (2x6)]
    
    # Sub-problem 1: A_left (rows 0-2, cols 0-1) * B_top (rows 0-1, cols 0-5)
    # A_left rows: 0,1,2; cols: 0,1
    # B_top rows: 0,1; cols: 0-5
    # Split A_left into top (row 0) and bot (rows 1,2)
    # Split B_top into left (cols 0-2) and right (cols 3-5)
    
    # A_top: row 0, cols 0,1 -> 1x2
    # A_bot: rows 1,2, cols 0,1 -> 2x2
    # B_left: rows 0,1, cols 0,1,2 -> 2x3
    # B_right: rows 0,1, cols 3,4,5 -> 2x3
    
    # A_top * B_left: 1x2 * 2x3 -> C[0, 0:3]
    add_1x2x2x3(0, [0, 1], [0, 1], [0, 1, 2], 0)
    
    # A_top * B_right: 1x2 * 2x3 -> C[0, 3:6]
    add_1x2x2x3(0, [0, 1], [0, 1], [3, 4, 5], 0)
    
    # A_bot * B_left: 2x2 * 2x3 -> C[1:3, 0:3]
    # Split B_left into B_left_left (2x1) and B_left_right (2x2)
    # A_bot * B_left_left: 2x2 * 2x1
    add_2x2x2x1([1, 2], [0, 1], [0, 1], [0], [1, 2], 0)
    # Wait, that's not right. Let me redo this.
    
    # Actually, for 2x2 * 2x3, I need to handle it as:
    # B is 2x3, split into B[:,0] (2x1) and B[:,1:3] (2x2)
    # A * B[:,0]: 2x2 * 2x1, rank 4
    # A * B[:,1:3]: 2x2 * 2x2, rank 7 (Strassen)
    
    # Let me redo the 2x2 * 2x3 helper
    def add_2x2x2x3(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add terms for 2x2 * 2x3.
        a_rows: 2 global row indices for A
        a_cols: 2 global column indices for A
        b_rows: 2 global row indices for B
        b_cols: 3 global column indices for B
        c_rows: 2 global row indices for C
        c_cols: 3 global column indices for C
        """
        # First part: A * B[:,0] (2x2 * 2x1)
        # C[:,0] = A * B[:,0]
        for i in range(2):
            for k in range(2):
                add_term(a_rows[i] * m + a_cols[k], b_rows[k] * p + b_cols[0], c_rows[i] * n + c_cols[0], 1)
        
        # Second part: A * B[:,1:3] (2x2 * 2x2) using Strassen
        add_2x2x2x2_strassen(a_rows, a_cols, b_rows, [b_cols[1], b_cols[2]], c_rows, [c_cols[1], c_cols[2]])
    
    # Clear terms and rebuild properly
    terms = []
    
    # Sub-problem 1: A_left (rows 0-2, cols 0-1) * B_top (rows 0-1, cols 0-5)
    
    # A_top * B_left: 1x2 * 2x3 -> C[0, 0:3]
    add_1x2x2x3(0, [0, 1], [0, 1], [0, 1, 2], 0)
    
    # A_top * B_right: 1x2 * 2x3 -> C[0, 3:6]
    add_1x2x2x3(0, [0, 1], [0, 1], [3, 4, 5], 0)
    
    # A_bot * B_left: 2x2 * 2x3 -> C[1:3, 0:3]
    add_2x2x2x3([1, 2], [0, 1], [0, 1], [0, 1, 2], [1, 2], [0, 1, 2])
    
    # A_bot * B_right: 2x2 * 2x3 -> C[1:3, 3:6]
    add_2x2x2x3([1, 2], [0, 1], [0, 1], [3, 4, 5], [1, 2], [3, 4, 5])
    
    # Sub-problem 2: A_right (rows 0-2, cols 2-3) * B_bottom (rows 2-3, cols 0-5)
    
    # A_top_right * B_left_bottom: 1x2 * 2x3 -> C[0, 0:3]
    add_1x2x2x3(0, [2, 3], [2, 3], [0, 1, 2], 0)
    
    # A_top_right * B_right_bottom: 1x2 * 2x3 -> C[0, 3:6]
    add_1x2x2x3(0, [2, 3], [2, 3], [3, 4, 5], 0)
    
    # A_bot_right * B_left_bottom: 2x2 * 2x3 -> C[1:3, 0:3]
    add_2x2x2x3([1, 2], [2, 3], [2, 3], [0, 1, 2], [1, 2], [0, 1, 2])
    
    # A_bot_right * B_right_bottom: 2x2 * 2x3 -> C[1:3, 3:6]
    add_2x2x2x3([1, 2], [2, 3], [2, 3], [3, 4, 5], [1, 2], [3, 4, 5])
    
    # Now build U, V, W from terms
    rank = len(terms)
    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))
    
    for r, (u_pos, v_pos, w_pos, coeff) in enumerate(terms):
        U[u_pos, r] = coeff
        V[v_pos, r] = 1
        W[w_pos, r] = 1
    
    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((U_dim, rank_school))
        V = np.zeros((V_dim, rank_school))
        W = np.zeros((W_dim, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    return {"U": to_integer_list(U), "V": to_integer_list(V), "W": to_integer_list(W)}
# EVOLVE-BLOCK-END

"""Matrix multiplication 3x4x5 with low-rank decomposition using Strassen."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x5 matrix.
    
    The output factor W follows the trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Total rank estimate: 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # shape (12, 56)
    V = [[0] * rank for _ in range(m * p)]  # shape (20, 56)
    W = [[0] * rank for _ in range(p * n)]  # shape (15, 56)
    
    column = 0
    
    # Block decomposition: A = [A1 A2], B = [B1; B2]
    # A1: rows 0-2, cols 0-1 (3x2)
    # A2: rows 0-2, cols 2-3 (3x2)
    # B1: rows 0-1, cols 0-4 (2x5)
    # B2: rows 2-3, cols 0-4 (2x5)
    # C = A1*B1 + A2*B2
    
    # For each block multiplication A_block * B_block:
    # Use Strassen for 2x2x2 sub-blocks
    
    def add_block_u(u_idx_start, u_idx_end, col_offset, values):
        """Add values to U[u_idx_start:u_idx_end][col_offset]"""
        for i in range(u_idx_start, u_idx_end):
            U[i][col_offset] = values[i - u_idx_start]
    
    def add_block_v(v_idx_start, v_idx_end, col_offset, values):
        """Add values to V[v_idx_start:v_idx_end][col_offset]"""
        for i in range(v_idx_start, v_idx_end):
            V[i][col_offset] = values[i - v_idx_start]
    
    def add_block_w(w_idx_start, w_idx_end, col_offset, values):
        """Add values to W[w_idx_start:w_idx_end][col_offset]"""
        for i in range(w_idx_start, w_idx_end):
            W[i][col_offset] = values[i - w_idx_start]
    
    def strassen_2x2x2(u_offset, v_offset, w_offset, u_indices, v_indices, w_indices):
        """
        Add Strassen decomposition for 2x2x2 block.
        u_indices: list of 4 U indices (for A11, A12, A21, A22)
        v_indices: list of 4 V indices (for B11, B12, B21, B22)
        w_indices: list of 4 W indices (for C11, C12, C21, C22)
        Returns new column offset.
        """
        nonlocal column
        col = column
        
        # M1 = (A11 + A22)(B11 + B22)
        add_block_u(u_indices[0], u_indices[0]+1, col, [1])
        add_block_u(u_indices[3], u_indices[3]+1, col, [1])
        add_block_v(v_indices[0], v_indices[0]+1, col, [1])
        add_block_v(v_indices[3], v_indices[3]+1, col, [1])
        add_block_w(w_indices[0], w_indices[0]+1, col, [1])
        add_block_w(w_indices[3], w_indices[3]+1, col, [1])
        column += 1
        
        # M2 = (A21 + A22)B11
        add_block_u(u_indices[2], u_indices[2]+1, col, [1])
        add_block_u(u_indices[3], u_indices[3]+1, col, [1])
        add_block_v(v_indices[0], v_indices[0]+1, col, [1])
        add_block_w(w_indices[2], w_indices[2]+1, col, [1])
        add_block_w(w_indices[3], w_indices[3]+1, col, [1])
        column += 1
        
        # M3 = A11(B21 - B11)
        add_block_u(u_indices[0], u_indices[0]+1, col, [1])
        add_block_v(v_indices[2], v_indices[2]+1, col, [1])
        add_block_v(v_indices[0], v_indices[0]+1, col, [-1])
        add_block_w(w_indices[0], w_indices[0]+1, col, [1])
        add_block_w(w_indices[1], w_indices[1]+1, col, [1])
        column += 1
        
        # M4 = A22(B12 - B22)
        add_block_u(u_indices[3], u_indices[3]+1, col, [1])
        add_block_v(v_indices[1], v_indices[1]+1, col, [1])
        add_block_v(v_indices[3], v_indices[3]+1, col, [-1])
        add_block_w(w_indices[2], w_indices[2]+1, col, [1])
        add_block_w(w_indices[3], w_indices[3]+1, col, [1])
        column += 1
        
        # M5 = (A11 + A12)B22
        add_block_u(u_indices[0], u_indices[0]+1, col, [1])
        add_block_u(u_indices[1], u_indices[1]+1, col, [1])
        add_block_v(v_indices[3], v_indices[3]+1, col, [1])
        add_block_w(w_indices[1], w_indices[1]+1, col, [1])
        add_block_w(w_indices[3], w_indices[3]+1, col, [1])
        column += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        add_block_u(u_indices[2], u_indices[2]+1, col, [1])
        add_block_u(u_indices[0], u_indices[0]+1, col, [-1])
        add_block_v(v_indices[0], v_indices[0]+1, col, [1])
        add_block_v(v_indices[1], v_indices[1]+1, col, [1])
        add_block_w(w_indices[0], w_indices[0]+1, col, [1])
        add_block_w(w_indices[2], w_indices[2]+1, col, [1])
        column += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        add_block_u(u_indices[1], u_indices[1]+1, col, [1])
        add_block_u(u_indices[3], u_indices[3]+1, col, [-1])
        add_block_v(v_indices[2], v_indices[2]+1, col, [1])
        add_block_v(v_indices[3], v_indices[3]+1, col, [1])
        add_block_w(w_indices[1], w_indices[1]+1, col, [1])
        add_block_w(w_indices[3], w_indices[3]+1, col, [1])
        column += 1
        
        return column
    
    def schoolbook_1x2x5(u_idx, v_indices, w_indices, col_offset):
        """
        Schoolbook for 1x2x5 block.
        u_idx: single U index
        v_indices: 2 V indices
        w_indices: 5 W indices
        """
        nonlocal column
        col = col_offset
        for j in range(2):
            for k in range(5):
                add_block_u(u_idx, u_idx+1, col, [1])
                add_block_v(v_indices[j], v_indices[j]+1, col, [1])
                add_block_w(w_indices[k], w_indices[k]+1, col, [1])
                column += 1
        return column
    
    # Block 1: A1 * B1 (3x2 * 2x5)
    # Split 5 into 4+1
    # First 4: use Strassen for 2x2x4 (split into two 2x2x2)
    # Last 1: schoolbook
    
    # A1 rows: 0-2, cols 0-1
    # B1 rows: 0-1, cols 0-4
    # C1 rows: 0-2, cols 0-4
    
    # For rows 0-1 of A1 (2x2x5 part):
    # Split into 2x2x4 + 2x2x1
    # 2x2x4: split into two 2x2x2
    # 2x2x1: schoolbook
    
    # U indices for A1 rows 0-1: u_idx = 0*2 + 0 = 0, u_idx = 0*2 + 1 = 1
    # U indices for A1 row 2: u_idx = 2*2 + 0 = 4, u_idx = 2*2 + 1 = 5
    # V indices for B1: v_idx = 0*5 + 0 = 0, v_idx = 0*5 + 1 = 1, ..., v_idx = 1*5 + 4 = 9
    # W indices for C1: w_idx = 0*3 + 0 = 0, w_idx = 0*3 + 1 = 1, w_idx = 0*3 + 2 = 2, w_idx = 1*3 + 0 = 3, ..., w_idx = 4*3 + 2 = 14
    
    # 2x2x2 block 1: cols 0-1 of B1
    # u_indices: [0, 1, 2, 3] (A11=0, A12=1, A21=2, A22=3) - wait, A1 is 3x2, so rows 0-1 of A1 are u_idx 0-3
    # Actually, A1 is rows 0-2, cols 0-1. So for rows 0-1 of A1: u_idx = 0,1 (row 0, col 0), u_idx = 2,3 (row 1, col 0), u_idx = 4,5 (row 2, col 0)... no.
    # A is 3x4. A[i*m + j]. m=4.
    # A1 is cols 0-1. So A1[i][j] = A[i*4 + j] for j in 0-1.
    # U index for A[i][j] is i*4 + j.
    # A1 row 0: u_idx 0,1
    # A1 row 1: u_idx 4,5
    # A1 row 2: u_idx 8,9
    # B1 is rows 0-1, cols 0-4. B[j][k] = B[j*5 + k].
    # B1 row 0: v_idx 0-4
    # B1 row 1: v_idx 5-9
    # C is 3x5. C[i][k] = C[k*3 + i] (W convention).
    # C1 row 0: w_idx 0,3,6,9,12
    # C1 row 1: w_idx 1,4,7,10,13
    # C1 row 2: w_idx 2,5,8,11,14
    
    # 2x2x2 block 1: A1 rows 0-1, cols 0-1; B1 cols 0-1
    # u_indices: [0, 1, 4, 5] (A11=0, A12=1, A21=4, A22=5)
    # v_indices: [0, 1, 5, 6] (B11=0, B12=1, B21=5, B22=6)
    # w_indices: [0, 1, 3, 4] (C11=0, C12=1, C21=3, C22=4)
    strassen_2x2x2(0, 0, 0, [0, 1, 4, 5], [0, 1, 5, 6], [0, 1, 3, 4])
    
    # 2x2x2 block 2: A1 rows 0-1, cols 0-1; B1 cols 2-3
    # u_indices: [0, 1, 4, 5]
    # v_indices: [2, 3, 7, 8]
    # w_indices: [0, 1, 3, 4] (but for cols 2-3, so w_idx 2,5,8,11)
    # Actually, w_indices should be for the output positions
    # C1 row 0, col 2: w_idx 2; col 3: w_idx 5
    # C1 row 1, col 2: w_idx 8; col 3: w_idx 11
    # So w_indices: [2, 8, 5, 11] - wait, C11, C12, C21, C22 for this block
    # C11: row 0, col 2 -> w_idx 2
    # C12: row 0, col 3 -> w_idx 5
    # C21: row 1, col 2 -> w_idx 8
    # C22: row 1, col 3 -> w_idx 11
    strassen_2x2x2(0, 0, 0, [0, 1, 4, 5], [2, 3, 7, 8], [2, 5, 8, 11])
    
    # 2x2x1 block: A1 rows 0-1, cols 0-1; B1 col 4
    # u_indices: [0, 1, 4, 5]
    # v_indices: [4, 9]
    # w_indices: [0, 3] (row 0 col 4 -> w_idx 12, row 1 col 4 -> w_idx 13)
    # Wait, C1 row 0 col 4: w_idx = 4*3 + 0 = 12
    # C1 row 1 col 4: w_idx = 4*3 + 1 = 13
    # Schoolbook for 2x2x1: 4 terms
    for j in range(2):
        for k in range(1):
            add_block_u(0, 1, column, [1])
            add_block_v(4, 5, column, [1])
            add_block_w(12, 13, column, [1])
            column += 1
            add_block_u(4, 5, column, [1])
            add_block_v(9, 10, column, [1])
            add_block_w(13, 14, column, [1])
            column += 1
    
    # Now handle A1 row 2 (1x2x5)
    schoolbook_1x2x5(8, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 0)
    
    # Block 2: A2 * B2 (3x2 * 2x5)
    # A2 is rows 0-2, cols 2-3
    # U indices: A[i*4 + 2], A[i*4 + 3]
    # Row 0: u_idx 2, 3
    # Row 1: u_idx 6, 7
    # Row 2: u_idx 10, 11
    # B2 is rows 2-3, cols 0-4
    # V indices: B[2*5 + k] = 10+k, B[3*5 + k] = 15+k
    # C2 is rows 0-2, cols 0-4 (same positions as C1)
    
    # 2x2x2 block 1: A2 rows 0-1, cols 2-3; B2 cols 0-1
    strassen_2x2x2(0, 0, 0, [2, 3, 6, 7], [10, 11, 15, 16], [0, 1, 3, 4])
    
    # 2x2x2 block 2: A2 rows 0-1, cols 2-3; B2 cols 2-3
    strassen_2x2x2(0, 0, 0, [2, 3, 6, 7], [12, 13, 17, 18], [2, 5, 8, 11])
    
    # 2x2x1 block: A2 rows 0-1, cols 2-3; B2 col 4
    for j in range(2):
        for k in range(1):
            add_block_u(2, 3, column, [1])
            add_block_v(14, 15, column, [1])
            add_block_w(12, 13, column, [1])
            column += 1
            add_block_u(6, 7, column, [1])
            add_block_v(19, 20, column, [1])
            add_block_w(13, 14, column, [1])
            column += 1
    
    # A2 row 2 (1x2x5)
    schoolbook_1x2x5(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 0)
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Initialize U, V, W as lists of lists of zeros
    # We don't know the rank upfront exactly if we build dynamically, 
    # but we calculated 69. Let's use a dynamic list and convert to 2D array or just keep as list of lists.
    # The interface expects U, V, W to be factor matrices of shapes (n*m, rank), etc.
    # Returning a list of lists is usually accepted if it can be interpreted as such.
    
    U = []
    V = []
    W = []
    
    def add_term(u_data, v_data, w_data):
        """
        Adds one rank-1 term.
        u_data: list of (row_idx, value)
        v_data: list of (row_idx, value)
        w_data: list of (row_idx, value)
        """
        # Append to each matrix
        # We need to know the current rank to align columns.
        # Simpler: build lists of lists for each row.
        pass # Re-implementing strategy: Build column by column.

    # Better strategy: Pre-allocate or build lists of values for each row.
    # Let's build a list of columns, then transpose.
    cols_U = []
    cols_V = []
    cols_W = []
    
    def add_term(u_entries, v_entries, w_entries):
        # u_entries: dict {row: val} or list of tuples
        # Convert to dict for easy merging if we were merging, but we just append a column.
        col_u = [0] * (n * m)
        col_v = [0] * (m * p)
        col_w = [0] * (p * n)
        
        for idx, val in u_entries:
            col_u[idx] = val
        for idx, val in v_entries:
            col_v[idx] = val
        for idx, val in w_entries:
            col_w[idx] = val
            
        cols_U.append(col_u)
        cols_V.append(col_v)
        cols_W.append(col_w)
        
    def get_u_idx(i, j):
        return i * m + j
        
    def get_v_idx(j, k):
        return j * p + k
        
    def get_w_idx(i, k):
        # W[k*n + i, r]
        return k * n + i
        
    def add_strassen(i_A, j_A, i_B, j_B, i_C, j_C):
        # A block starts at i_A, j_A (2x2)
        # B block starts at i_B, j_B (2x2)
        # C block starts at i_C, j_C (2x2)
        
        # P1
        add_term(
            [(get_u_idx(i_A, j_A), 1), (get_u_idx(i_A+1, j_A+1), 1)],
            [(get_v_idx(i_B, j_B), 1), (get_v_idx(i_B+1, j_B+1), 1)],
            [(get_w_idx(i_C, j_C), 1), (get_w_idx(i_C+1, j_C+1), 1)]
        )
        # P2
        add_term(
            [(get_u_idx(i_A+1, j_A), 1), (get_u_idx(i_A+1, j_A+1), 1)],
            [(get_v_idx(i_B, j_B), 1)],
            [(get_w_idx(i_C+1, j_C), 1), (get_w_idx(i_C+1, j_C+1), -1)]
        )
        # P3
        add_term(
            [(get_u_idx(i_A, j_A), 1)],
            [(get_v_idx(i_B, j_B+1), 1), (get_v_idx(i_B+1, j_B+1), -1)],
            [(get_w_idx(i_C, j_C+1), 1), (get_w_idx(i_C+1, j_C+1), 1)]
        )
        # P4
        add_term(
            [(get_u_idx(i_A+1, j_A+1), 1)],
            [(get_v_idx(i_B+1, j_B+1), 1), (get_v_idx(i_B, j_B+1), -1)],
            [(get_w_idx(i_C, j_C), 1), (get_w_idx(i_C+1, j_C), 1)]
        )
        # P5
        add_term(
            [(get_u_idx(i_A, j_A), 1), (get_u_idx(i_A, j_A+1), 1)],
            [(get_v_idx(i_B+1, j_B+1), 1)],
            [(get_w_idx(i_C, j_C), -1), (get_w_idx(i_C, j_C+1), 1)]
        )
        # P6
        add_term(
            [(get_u_idx(i_A+1, j_A), 1), (get_u_idx(i_A, j_A), -1)],
            [(get_v_idx(i_B, j_B), 1), (get_v_idx(i_B, j_B+1), 1)],
            [(get_w_idx(i_C+1, j_C+1), 1)]
        )
        # P7
        add_term(
            [(get_u_idx(i_A, j_A+1), 1), (get_u_idx(i_A+1, j_A+1), -1)],
            [(get_v_idx(i_B+1, j_B), 1), (get_v_idx(i_B+1, j_B+1), 1)],
            [(get_w_idx(i_C, j_C), 1)]
        )

    def add_schoolblock(i_A_start, j_A_start, h_A, w_A, i_B_start, j_B_start, h_B, w_B, i_C_start, j_C_start, h_C, w_C):
        # Standard multiplication for rectangular blocks
        for di in range(h_A):
            for dj in range(w_A):
                i_A = i_A_start + di
                j_A = j_A_start + dj
                
                # This A element multiplies with B row j_A
                # B rows must match j_A. 
                # In our decomposition, we ensure B rows cover the range.
                # Specifically, for a term A[i_A, j_A] * B[j_A, k], C[i_A, k] += ...
                # We iterate over k in B's column range.
                for dk in range(w_B):
                    k = j_B_start + dk
                    # Check if j_A is within B's row range?
                    # In our specific splits, we ensure dimensions match.
                    # e.g., A is 2x1, B is 1x2. j_A is 0 or 1. B row 0 or 1.
                    # We assume the call is valid for the specific block structure.
                    
                    add_term(
                        [(get_u_idx(i_A, j_A), 1)],
                        [(get_v_idx(j_A, k), 1)],
                        [(get_w_idx(i_A, k), 1)]
                    )

    # Block 1: B cols 0-2 (3x3)
    # A TL (0,0) 2x2, TR (0,2) 2x1, BL (2,0) 1x2, BR (2,2) 1x1
    # B TL (0,0) 2x2, TR (0,2) 2x1, BL (2,0) 1x2, BR (2,2) 1x1
    # C TL (0,0) 2x2, TR (0,2) 2x1, BL (2,0) 1x2, BR (2,2) 1x1
    
    # C_TL = A_TL * B_TL (Strassen) + A_TR * B_BL (School)
    add_strassen(0, 0, 0, 0, 0, 0)
    add_schoolblock(0, 2, 2, 1, 2, 0, 1, 2, 0, 0, 2, 2)
    
    # C_TR = A_TL * B_TR (School) + A_TR * B_BR (School)
    add_schoolblock(0, 0, 2, 2, 0, 2, 2, 1, 0, 2, 2, 1)
    add_schoolblock(0, 2, 2, 1, 2, 2, 1, 1, 0, 2, 2, 1)
    
    # C_BL = A_BL * B_TL (School) + A_BR * B_BL (School)
    add_schoolblock(2, 0, 1, 2, 0, 0, 2, 2, 2, 0, 1, 2)
    add_schoolblock(2, 2, 1, 1, 2, 0, 1, 2, 2, 0, 1, 2)
    
    # C_BR = A_BL * B_TR (School) + A_BR * B_BR (School)
    add_schoolblock(2, 0, 1, 2, 0, 2, 2, 1, 2, 2, 1, 1)
    add_schoolblock(2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1)
    
    # Block 2: B cols 3-5 (3x3)
    # Same as Block 1, but B cols offset by 3, C cols offset by 3
    
    add_strassen(0, 0, 0, 3, 0, 3)
    add_schoolblock(0, 2, 2, 1, 2, 3, 1, 2, 0, 3, 2, 2)
    
    add_schoolblock(0, 0, 2, 2, 0, 5, 2, 1, 0, 5, 2, 1)
    add_schoolblock(0, 2, 2, 1, 2, 5, 1, 1, 0, 5, 2, 1)
    
    add_schoolblock(2, 0, 1, 2, 0, 3, 2, 2, 2, 3, 1, 2)
    add_schoolblock(2, 2, 1, 1, 2, 3, 1, 2, 2, 3, 1, 2)
    
    add_schoolblock(2, 0, 1, 2, 0, 5, 2, 1, 2, 5, 1, 1)
    add_schoolblock(2, 2, 1, 1, 2, 5, 1, 1, 2, 5, 1, 1)

    # Block 3: B cols 6-7 (3x2)
    # A same.
    # B TL (0,6) 2x2, BL (2,6) 1x2
    # C TL (0,6) 2x2, BL (2,6) 1x2
    
    add_strassen(0, 0, 0, 6, 0, 6)
    add_schoolblock(0, 2, 2, 1, 2, 6, 1, 2, 0, 6, 2, 2)
    
    add_schoolblock(2, 0, 1, 2, 0, 6, 2, 2, 2, 6, 1, 2)
    add_schoolblock(2, 2, 1, 1, 2, 6, 1, 2, 2, 6, 1, 2)

    # Convert cols to rows for output
    rank = len(cols_U)
    U = [[cols_U[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[cols_V[r][i] for r in range(rank)] for i in range(m*p)]
    W = [[cols_W[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy:
    Split the 3x4 matrix A into top 2 rows and bottom 1 row.
    Split the 4x8 matrix B into 4 column blocks of width 2.
    
    For each column block of B (width 2):
    1. Top part of C (2x2): Computed as A_top (2x4) * B_block (4x2).
       This is equivalent to A_top_left (2x2) * B_top (2x2) + A_top_right (2x2) * B_bottom (2x2).
       We use Strassen's algorithm (rank 7) for each 2x2x2 multiplication.
       Total rank per block = 7 + 7 = 14.
       
    2. Bottom part of C (1x2): Computed as A_bottom (1x4) * B_block (4x2).
       This is equivalent to A_bottom_left (1x2) * B_top (2x2) + A_bottom_right (1x2) * B_bottom (2x2).
       We use schoolbook (rank 4) for each 1x2x2 multiplication.
       Total rank per block = 4 + 4 = 8.
       
    Total rank = 4 blocks * (14 + 8) = 88.
    """
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (12, R)
    # V: (m*p, rank) = (32, R)
    # W: (p*n, rank) = (24, R)
    
    total_rank = 88
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
        A is at [a_row_offset:a_row_offset+2, a_col_offset:a_col_offset+2]
        B is at [b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2]
        C is at [c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]
        """
        nonlocal col
        
        # Indices for A (2x2)
        a00 = a_row_offset * m + a_col_offset
        a01 = a_row_offset * m + a_col_offset + 1
        a10 = (a_row_offset + 1) * m + a_col_offset
        a11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        # Indices for B (2x2)
        b00 = b_row_offset * p + b_col_offset
        b01 = b_row_offset * p + b_col_offset + 1
        b10 = (b_row_offset + 1) * p + b_col_offset
        b11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        # Indices for C (2x2) - W[k*n + i]
        # C is 2x2.
        # c00 -> row 0, col 0 -> W[c_col_offset * n + c_row_offset]
        # c01 -> row 0, col 1 -> W[(c_col_offset+1) * n + c_row_offset]
        # c10 -> row 1, col 0 -> W[c_col_offset * n + (c_row_offset+1)]
        # c11 -> row 1, col 1 -> W[(c_col_offset+1) * n + (c_row_offset+1)]
        
        c00 = c_col_offset * n + c_row_offset
        c01 = (c_col_offset + 1) * n + c_row_offset
        c10 = c_col_offset * n + (c_row_offset + 1)
        c11 = (c_col_offset + 1) * n + (c_row_offset + 1)
        
        # M1 = A00 * (B00 + B11)
        U[a00][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M3 = A00 * (B10 - B00)
        U[a00][col] += 1
        V[b10][col] += 1
        V[b00][col] += -1
        W[c10][col] += 1
        col += 1
        
        # M4 = A11 * (B01 - B11)
        U[a11][col] += 1
        V[b01][col] += 1
        V[b11][col] += -1
        W[c01][col] += 1
        col += 1
        
        # M5 = (A00 + A11) * (B00 + B11)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c01][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[a10][col] += 1
        U[a00][col] += -1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c00][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[a01][col] += 1
        U[a11][col] += -1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c01][col] += 1
        W[c10][col] += 1
        col += 1

    def add_schoolbook_1x2x2(a_row, a_col_start, b_row_start, b_col_start, c_row, c_col_start):
        """
        Adds a rank-4 schoolbook decomposition for a 1x2 * 2x2 block.
        A is 1x2 at [a_row, a_col_start:a_col_start+2]
        B is 2x2 at [b_row_start:b_row_start+2, b_col_start:b_col_start+2]
        C is 1x2 at [c_row, c_col_start:c_col_start+2]
        """
        nonlocal col
        
        # A indices
        a0 = a_row * m + a_col_start
        a1 = a_row * m + a_col_start + 1
        
        # B indices
        b00 = b_row_start * p + b_col_start
        b01 = b_row_start * p + b_col_start + 1
        b10 = (b_row_start + 1) * p + b_col_start
        b11 = (b_row_start + 1) * p + b_col_start + 1
        
        # C indices
        c0 = c_col_start * n + c_row
        c1 = (c_col_start + 1) * n + c_row
        
        # C0 = a0*b00 + a1*b10
        # Term 1: a0*b00
        U[a0][col] += 1
        V[b00][col] += 1
        W[c0][col] += 1
        col += 1
        # Term 2: a1*b10
        U[a1][col] += 1
        V[b10][col] += 1
        W[c0][col] += 1
        col += 1
        
        # C1 = a0*b01 + a1*b11
        # Term 3: a0*b01
        U[a0][col] += 1
        V[b01][col] += 1
        W[c1][col] += 1
        col += 1
        # Term 4: a1*b11
        U[a1][col] += 1
        V[b11][col] += 1
        W[c1][col] += 1
        col += 1

    # Iterate over the 4 column blocks of B (and corresponding C columns)
    for block_idx in range(4):
        b_col_offset = block_idx * 2
        c_col_offset = block_idx * 2
        
        # Top part of C (rows 0, 1)
        # A rows 0, 1. Cols 0,1 (left) and 2,3 (right)
        # B rows 0,1 (top) and 2,3 (bottom). Cols b_col_offset, b_col_offset+1
        
        # 1. A_top_left * B_top
        add_strassen_2x2(
            a_row_offset=0, a_col_offset=0, 
            b_row_offset=0, b_col_offset=b_col_offset,
            c_row_offset=0, c_col_offset=c_col_offset
        )
        
        # 2. A_top_right * B_bottom
        add_strassen_2x2(
            a_row_offset=0, a_col_offset=2,
            b_row_offset=2, b_col_offset=b_col_offset,
            c_row_offset=0, c_col_offset=c_col_offset
        )
        
        # Bottom part of C (row 2)
        # A row 2. Cols 0,1 (left) and 2,3 (right)
        # B rows 0,1 (top) and 2,3 (bottom). Cols b_col_offset, b_col_offset+1
        
        # 3. A_bottom_left * B_top
        add_schoolbook_1x2x2(
            a_row=2, a_col_start=0,
            b_row_start=0, b_col_start=b_col_offset,
            c_row=2, c_col_start=c_col_offset
        )
        
        # 4. A_bottom_right * B_bottom
        add_schoolbook_1x2x2(
            a_row=2, a_col_start=2,
            b_row_start=2, b_col_start=b_col_offset,
            c_row=2, c_col_start=c_col_offset
        )
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

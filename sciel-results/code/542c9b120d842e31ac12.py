"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # We decompose the multiplication into blocks.
    # A is 2x4, B is 4x5.
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B into B_top (2x5) and B_bottom (2x5).
    # B_top is split into B11 (2x2), B12 (2x2), B13 (2x1).
    # B_bottom is split into B21 (2x2), B22 (2x2), B23 (2x1).
    
    # C = A1 * B_top + A2 * B_bottom
    # A1 * B_top = A1*B11 + A1*B12 + A1*B13
    # A2 * B_bottom = A2*B21 + A2*B22 + A2*B23
    
    # Total rank = 3 * (rank 2x2x2) + 2 * (rank 2x2x1) + 3 * (rank 2x2x2) + 2 * (rank 2x2x1)
    # Using schoolbook: rank 2x2x2 is 8, rank 2x2x1 is 2.
    # Total = 3*8 + 2*2 + 3*8 + 2*2 = 24 + 4 + 24 + 4 = 56?
    # Wait.
    # A1*B11 (2x2x2) -> 8
    # A1*B12 (2x2x2) -> 8
    # A1*B13 (2x2x1) -> 2
    # Sum for A1: 18.
    # Sum for A2: 18.
    # Total 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 8 x 36
    V = [[0] * rank for _ in range(m * p)] # 20 x 36
    W = [[0] * rank for _ in range(p * n)] # 10 x 36
    
    col = 0
    
    # Helper to add a 2x2x2 schoolbook block
    def add_2x2x2(a_row_offset, b_row_offset, c_col_offset, c_row_offset):
        # a_row_offset: row index in U for A block (0 or 2)
        # b_row_offset: row index in V for B block (0 or 2)
        # c_col_offset: col index in W for C block (0, 2, or 4)
        # c_row_offset: row index in W for C block (0 or 1)
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A[i, a_row_offset+j] * B[b_row_offset+j, c_col_offset+k] -> C[i, c_col_offset+k]
                    # U index: i*m + (a_row_offset+j)
                    # V index: (b_row_offset+j)*p + (c_col_offset+k)
                    # W index: (c_col_offset+k)*n + i
                    u_idx = i * m + (a_row_offset + j)
                    v_idx = (b_row_offset + j) * p + (c_col_offset + k)
                    w_idx = (c_col_offset + k) * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1

    # Helper to add a 2x2x1 schoolbook block
    def add_2x2x1(a_row_offset, b_row_offset, c_col_offset):
        # B block is 2x1, so k=0 only.
        nonlocal col
        for i in range(2):
            for j in range(2):
                # A[i, a_row_offset+j] * B[b_row_offset+j, c_col_offset] -> C[i, c_col_offset]
                u_idx = i * m + (a_row_offset + j)
                v_idx = (b_row_offset + j) * p + c_col_offset
                w_idx = c_col_offset * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    # A1 is rows 0-1 of A (cols 0-1 of A? No, A is 2x4. A1 is cols 0-1).
    # A indices: i in 0..1, j in 0..3.
    # A1: j in 0..1. A2: j in 2..3.
    
    # B1 (top): rows 0-1 of B. B2 (bottom): rows 2-3 of B.
    # B indices: j in 0..3, k in 0..4.
    # B11: j 0-1, k 0-1.
    # B12: j 0-1, k 2-3.
    # B13: j 0-1, k 4.
    
    # B21: j 2-3, k 0-1.
    # B22: j 2-3, k 2-3.
    # B23: j 2-3, k 4.
    
    # C indices: i in 0..1, k in 0..4.
    
    # Block 1: A1 * B11
    # A1: a_row_offset=0 (j=0,1)
    # B11: b_row_offset=0 (j=0,1), c_col_offset=0 (k=0,1)
    add_2x2x2(0, 0, 0)
    
    # Block 2: A1 * B12
    # A1: a_row_offset=0
    # B12: b_row_offset=0, c_col_offset=2 (k=2,3)
    add_2x2x2(0, 0, 2)
    
    # Block 3: A1 * B13
    # A1: a_row_offset=0
    # B13: b_row_offset=0, c_col_offset=4 (k=4)
    add_2x2x1(0, 0, 4)
    
    # Block 4: A2 * B21
    # A2: a_row_offset=2 (j=2,3)
    # B21: b_row_offset=2 (j=2,3), c_col_offset=0
    add_2x2x2(2, 2, 0)
    
    # Block 5: A2 * B22
    # A2: a_row_offset=2
    # B22: b_row_offset=2, c_col_offset=2
    add_2x2x2(2, 2, 2)
    
    # Block 6: A2 * B23
    # A2: a_row_offset=2
    # B23: b_row_offset=2, c_col_offset=4
    add_2x2x1(2, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

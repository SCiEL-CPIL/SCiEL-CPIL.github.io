"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decomposition strategy:
    # Split B into B1 (4x2) and B2 (4x3).
    # C = A B1 + A B2.
    # C1 = A B1 (2x2). Split A into A1 (2x2), A2 (2x2).
    # C1 = A1 B11 + A2 B21. Use Strassen for each (rank 7).
    # C2 = A B2 (2x3). Split B2 into B21 (4x1) and B22 (4x2).
    # C21 = A B21 (2x1). Schoolbook (rank 8).
    # C22 = A B22 (2x2). Split A into A1, A2.
    # C22 = A1 B12 + A2 B22. Use Strassen for each (rank 7).
    # Total rank = 7 + 7 + 8 + 7 + 7 = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # A is 2x2 at (a_row_offset, a_col_offset)
        # B is 2x2 at (b_row_offset, b_col_offset)
        # C is 2x2 at (c_row_offset, c_col_offset)
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # A elements:
        # a00: (a_row_offset, a_col_offset) -> U[(a_row_offset)*m + a_col_offset]
        # a01: (a_row_offset, a_col_offset+1) -> U[(a_row_offset)*m + a_col_offset+1]
        # a10: (a_row_offset+1, a_col_offset) -> U[(a_row_offset+1)*m + a_col_offset]
        # a11: (a_row_offset+1, a_col_offset+1) -> U[(a_row_offset+1)*m + a_col_offset+1]
        
        # B elements:
        # b00: (b_row_offset, b_col_offset) -> V[(b_row_offset)*p + b_col_offset]
        # b01: (b_row_offset, b_col_offset+1) -> V[(b_row_offset)*p + b_col_offset+1]
        # b10: (b_row_offset+1, b_col_offset) -> V[(b_row_offset+1)*p + b_col_offset]
        # b11: (b_row_offset+1, b_col_offset+1) -> V[(b_row_offset+1)*p + b_col_offset+1]
        
        # C elements:
        # c00: (c_row_offset, c_col_offset) -> W[(c_col_offset)*n + c_row_offset]
        # c01: (c_row_offset, c_col_offset+1) -> W[(c_col_offset+1)*n + c_row_offset]
        # c10: (c_row_offset+1, c_col_offset) -> W[(c_col_offset)*n + c_row_offset+1]
        # c11: (c_row_offset+1, c_col_offset+1) -> W[(c_col_offset+1)*n + c_row_offset+1]
        
        # Strassen terms:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v_idx_00 = b_row_offset * p + b_col_offset
        v_idx_01 = b_row_offset * p + b_col_offset + 1
        v_idx_10 = (b_row_offset + 1) * p + b_col_offset
        v_idx_11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        w_idx_00 = c_col_offset * n + c_row_offset
        w_idx_01 = (c_col_offset + 1) * n + c_row_offset
        w_idx_10 = c_col_offset * n + c_row_offset + 1
        w_idx_11 = (c_col_offset + 1) * n + c_row_offset + 1
        
        # M1
        U[u_idx_00][col] += 1
        U[u_idx_11][col] += 1
        V[v_idx_00][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M2
        U[u_idx_10][col] += 1
        U[u_idx_11][col] += 1
        V[v_idx_00][col] += 1
        W[w_idx_10][col] += 1
        W[w_idx_11][col] -= 1
        col += 1
        
        # M3
        U[u_idx_00][col] += 1
        V[v_idx_01][col] += 1
        V[v_idx_11][col] -= 1
        W[w_idx_01][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M4
        U[u_idx_11][col] += 1
        V[v_idx_10][col] += 1
        V[v_idx_00][col] -= 1
        W[w_idx_00][col] += 1
        W[w_idx_10][col] += 1
        col += 1
        
        # M5
        U[u_idx_00][col] += 1
        U[u_idx_01][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] -= 1
        W[w_idx_01][col] += 1
        col += 1
        
        # M6
        U[u_idx_10][col] += 1
        U[u_idx_00][col] -= 1
        V[v_idx_00][col] += 1
        V[v_idx_01][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M7
        U[u_idx_01][col] += 1
        U[u_idx_11][col] -= 1
        V[v_idx_10][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] += 1
        col += 1

    def add_schoolbook_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # A is 2x2 (rows a_row_offset..a_row_offset+1, cols a_col_offset..a_col_offset+1)
        # B is 2x1 (rows b_row_offset..b_row_offset+1, col b_col_offset)
        # C is 2x1 (rows c_row_offset..c_row_offset+1, col c_col_offset)
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v_idx_0 = b_row_offset * p + b_col_offset
        v_idx_1 = (b_row_offset + 1) * p + b_col_offset
        
        w_idx_0 = c_col_offset * n + c_row_offset
        w_idx_1 = c_col_offset * n + c_row_offset + 1
        
        # Term 1: A00 B0 -> C0
        U[u_idx_00][col] = 1
        V[v_idx_0][col] = 1
        W[w_idx_0][col] = 1
        col += 1
        
        # Term 2: A01 B1 -> C0
        U[u_idx_01][col] = 1
        V[v_idx_1][col] = 1
        W[w_idx_0][col] = 1
        col += 1
        
        # Term 3: A10 B0 -> C1
        U[u_idx_10][col] = 1
        V[v_idx_0][col] = 1
        W[w_idx_1][col] = 1
        col += 1
        
        # Term 4: A11 B1 -> C1
        U[u_idx_11][col] = 1
        V[v_idx_1][col] = 1
        W[w_idx_1][col] = 1
        col += 1

    # Part 1: C1 = A B1 (2x2)
    # B1 is cols 0,1 of B. Rows 0..3.
    # Split A into A1 (cols 0,1) and A2 (cols 2,3).
    # A1 is rows 0,1 cols 0,1. B11 is rows 0,1 cols 0,1.
    add_strassen(0, 0, 0, 0, 0, 0)
    # A2 is rows 0,1 cols 2,3. B21 is rows 2,3 cols 0,1.
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Part 2: C2 = A B2 (2x3)
    # B2 is cols 2,3,4 of B. Rows 0..3.
    # Split B2 into B21 (col 2) and B22 (cols 3,4).
    # C21 = A B21 (2x1).
    # B21 is rows 0..3, col 2.
    # Split A into A1 (cols 0,1) and A2 (cols 2,3).
    # A1 B21_top (2x1) + A2 B21_bot (2x1).
    # B21_top is rows 0,1 col 2.
    add_schoolbook_2x1(0, 0, 0, 2, 0, 2)
    # B21_bot is rows 2,3 col 2.
    add_schoolbook_2x1(0, 2, 2, 2, 0, 2)
    
    # C22 = A B22 (2x2).
    # B22 is cols 3,4. Rows 0..3.
    # Split A into A1 (cols 0,1) and A2 (cols 2,3).
    # A1 B22_top (2x2) + A2 B22_bot (2x2).
    # B22_top is rows 0,1 cols 3,4.
    add_strassen(0, 0, 0, 3, 0, 3)
    # B22_bot is rows 2,3 cols 3,4.
    add_strassen(0, 2, 2, 3, 0, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

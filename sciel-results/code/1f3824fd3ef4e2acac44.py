"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using a split of P=5 into 3 and 2.
    # Rank 2x4x3 = 2 * Rank(2x2x3) = 2 * 11 = 22 (using Strassen+Schoolbook for 2x2x3)
    # Rank 2x4x2 = 2 * Rank(2x2x2) = 2 * 7 = 14
    # Total Rank = 22 + 14 = 36.
    # This is the same rank but a different structural decomposition.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_2x2x3(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # A is 2x2, B is 2x3, C is 2x3.
        # Split B into B1 (2x2) and B2 (2x1).
        # C1 = A B1 (2x2) using Strassen (7).
        # C2 = A B2 (2x1) using Schoolbook (4).
        # Total 11.
        
        # Strassen part for cols 0,1 of B (local indices)
        add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
        
        # Schoolbook part for col 2 of B (local index)
        # B col 2 is at b_col_offset + 2
        # C col 2 is at c_col_offset + 2
        add_schoolbook_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset + 2, c_row_offset, c_col_offset + 2)

    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # Standard Strassen for 2x2x2
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
        
        U[u_idx_00][col] += 1; U[u_idx_11][col] += 1; V[v_idx_00][col] += 1; V[v_idx_11][col] += 1; W[w_idx_00][col] += 1; W[w_idx_11][col] += 1; col += 1
        U[u_idx_10][col] += 1; U[u_idx_11][col] += 1; V[v_idx_00][col] += 1; W[w_idx_10][col] += 1; W[w_idx_11][col] -= 1; col += 1
        U[u_idx_00][col] += 1; V[v_idx_01][col] += 1; V[v_idx_11][col] -= 1; W[w_idx_01][col] += 1; W[w_idx_11][col] += 1; col += 1
        U[u_idx_11][col] += 1; V[v_idx_10][col] += 1; V[v_idx_00][col] -= 1; W[w_idx_00][col] += 1; W[w_idx_10][col] += 1; col += 1
        U[u_idx_00][col] += 1; U[u_idx_01][col] += 1; V[v_idx_11][col] += 1; W[w_idx_00][col] -= 1; W[w_idx_01][col] += 1; col += 1
        U[u_idx_10][col] += 1; U[u_idx_00][col] -= 1; V[v_idx_00][col] += 1; V[v_idx_01][col] += 1; W[w_idx_11][col] += 1; col += 1
        U[u_idx_01][col] += 1; U[u_idx_11][col] -= 1; V[v_idx_10][col] += 1; V[v_idx_11][col] += 1; W[w_idx_00][col] += 1; col += 1

    def add_schoolbook_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v_idx_0 = b_row_offset * p + b_col_offset
        v_idx_1 = (b_row_offset + 1) * p + b_col_offset
        
        w_idx_0 = c_col_offset * n + c_row_offset
        w_idx_1 = c_col_offset * n + c_row_offset + 1
        
        U[u_idx_00][col] = 1; V[v_idx_0][col] = 1; W[w_idx_0][col] = 1; col += 1
        U[u_idx_01][col] = 1; V[v_idx_1][col] = 1; W[w_idx_0][col] = 1; col += 1
        U[u_idx_10][col] = 1; V[v_idx_0][col] = 1; W[w_idx_1][col] = 1; col += 1
        U[u_idx_11][col] = 1; V[v_idx_1][col] = 1; W[w_idx_1][col] = 1; col += 1

    # Split P=5 into P1=3 (cols 0-2) and P2=2 (cols 3-4)
    
    # Part 1: C1 = A B1 (2x3)
    # Split M=4 into M1=2 (rows 0-1) and M2=2 (rows 2-3)
    # A1 B11 (2x2x3) + A2 B12 (2x2x3)
    add_2x2x3(0, 0, 0, 0, 0, 0)
    add_2x2x3(0, 2, 2, 0, 0, 0)
    
    # Part 2: C2 = A B2 (2x2)
    # Split M=4 into M1=2 and M2=2
    # A1 B21 (2x2x2) + A2 B22 (2x2x2)
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

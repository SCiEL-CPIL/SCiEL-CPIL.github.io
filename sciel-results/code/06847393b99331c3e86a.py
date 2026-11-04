"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 2 and 3
    # Split p=7 into 3, 3, 1
    
    # Blocks:
    # A1: 3x2, A2: 3x3
    # B11: 2x3, B12: 2x3, B13: 2x1
    # B21: 3x3, B22: 3x3, B23: 3x1
    
    # We will build U, V, W by concatenating the schoolbook decompositions of each block.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(n_b, m_b, p_b, u_offset, v_offset, w_offset):
        """
        Adds a schoolbook decomposition for a block of size n_b x m_b x p_b.
        u_offset, v_offset, w_offset are the starting indices for the rows in U, V, W.
        """
        rank_b = n_b * m_b * p_b
        # U block: (n_b * m_b) x rank_b
        U_b = [[0] * rank_b for _ in range(n_b * m_b)]
        # V block: (m_b * p_b) x rank_b
        V_b = [[0] * rank_b for _ in range(m_b * p_b)]
        # W block: (p_b * n_b) x rank_b
        W_b = [[0] * rank_b for _ in range(p_b * n_b)]
        
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_b[i * m_b + j][col] = 1
                    V_b[j * p_b + k][col] = 1
                    W_b[k * n_b + i][col] = 1
                    col += 1
        return U_b, V_b, W_b

    # Block 1: A1 (3x2) * B11 (2x3)
    # n=3, m=2, p=3
    U1, V1, W1 = add_block(3, 2, 3, 0, 0, 0)
    U_list.append(U1)
    V_list.append(V1)
    W_list.append(W1)
    
    # Block 2: A1 (3x2) * B12 (2x3)
    # n=3, m=2, p=3
    U2, V2, W2 = add_block(3, 2, 3, 0, 0, 0)
    U_list.append(U2)
    V_list.append(V2)
    W_list.append(W2)
    
    # Block 3: A1 (3x2) * B13 (2x1)
    # n=3, m=2, p=1
    U3, V3, W3 = add_block(3, 2, 1, 0, 0, 0)
    U_list.append(U3)
    V_list.append(V3)
    W_list.append(W3)
    
    # Block 4: A2 (3x3) * B21 (3x3)
    # n=3, m=3, p=3
    U4, V4, W4 = add_block(3, 3, 3, 0, 0, 0)
    U_list.append(U4)
    V_list.append(V4)
    W_list.append(W4)
    
    # Block 5: A2 (3x3) * B22 (3x3)
    # n=3, m=3, p=3
    U5, V5, W5 = add_block(3, 3, 3, 0, 0, 0)
    U_list.append(U5)
    V_list.append(V5)
    W_list.append(W5)
    
    # Block 6: A2 (3x3) * B23 (3x1)
    # n=3, m=3, p=1
    U6, V6, W6 = add_block(3, 3, 1, 0, 0, 0)
    U_list.append(U6)
    V_list.append(V6)
    W_list.append(W6)
    
    # Now we need to assemble the full U, V, W matrices.
    # The rows of U correspond to (i, j) in the original A (3x5).
    # A1 is rows 0-2, cols 0-1. A2 is rows 0-2, cols 2-4.
    # So for a given i in 0..2:
    #   U rows for A1 are i*2 + j for j in 0..1.
    #   U rows for A2 are i*3 + j for j in 0..2? No, A2 is a 3x3 block.
    #   In the full U matrix, the row index is i*m + j.
    #   For A1 (cols 0,1), rows are i*5 + 0, i*5 + 1.
    #   For A2 (cols 2,3,4), rows are i*5 + 2, i*5 + 3, i*5 + 4.
    
    # Similarly for V (5x7):
    #   B11 (rows 0,1; cols 0,1,2): V rows 0*7+0..2, 1*7+0..2.
    #   B12 (rows 0,1; cols 3,4,5): V rows 0*7+3..5, 1*7+3..5.
    #   B13 (rows 0,1; cols 6): V rows 0*7+6, 1*7+6.
    #   B21 (rows 2,3,4; cols 0,1,2): V rows 2*7+0..2, 3*7+0..2, 4*7+0..2.
    #   B22 (rows 2,3,4; cols 3,4,5): V rows 2*7+3..5, 3*7+3..5, 4*7+3..5.
    #   B23 (rows 2,3,4; cols 6): V rows 2*7+6, 3*7+6, 4*7+6.
    
    # W (7x3):
    #   C11 (cols 0,1,2 of C; rows 0..2 of B11? No, W is indexed by k*n+i).
    #   W rows correspond to (k, i).
    #   For B11 (p=3), k is 0..2. i is 0..2.
    #   For B12 (p=3), k is 0..2 (local), global k is 3..5.
    #   For B13 (p=1), k is 0 (local), global k is 6.
    #   For B21 (p=3), k is 0..2 (local), global k is 0..2.
    #   For B22 (p=3), k is 0..2 (local), global k is 3..5.
    #   For B23 (p=1), k is 0 (local), global k is 6.

    # Total rank
    total_rank = 18 + 18 + 6 + 9 + 9 + 9
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to place block matrices into full matrices
    def place_block(U_b, V_b, W_b, u_rows, v_rows, w_rows, col_off):
        for r_u, row_u in enumerate(U_b):
            for c, val in enumerate(row_u):
                if val != 0:
                    U[u_rows[r_u]][col_off + c] += val
        for r_v, row_v in enumerate(V_b):
            for c, val in enumerate(row_v):
                if val != 0:
                    V[v_rows[r_v]][col_off + c] += val
        for r_w, row_w in enumerate(W_b):
            for c, val in enumerate(row_w):
                if val != 0:
                    W[w_rows[r_w]][col_off + c] += val

    # Block 1: A1 (3x2) * B11 (2x3)
    # U rows: A1 is cols 0,1 of A. i=0..2.
    # Local U row (i, j) -> Global U row i*5 + j.
    u_rows_1 = [i * 5 + j for i in range(3) for j in range(2)]
    # V rows: B11 is rows 0,1; cols 0,1,2 of B.
    # Local V row (j, k) -> Global V row j*7 + k.
    v_rows_1 = [j * 7 + k for j in range(2) for k in range(3)]
    # W rows: C11 is cols 0,1,2 of C; rows 0..2.
    # Local W row (k, i) -> Global W row k*3 + i.
    w_rows_1 = [k * 3 + i for k in range(3) for i in range(3)]
    place_block(U1, V1, W1, u_rows_1, v_rows_1, w_rows_1, col_offset)
    col_offset += 18

    # Block 2: A1 (3x2) * B12 (2x3)
    # U rows: Same as Block 1.
    u_rows_2 = [i * 5 + j for i in range(3) for j in range(2)]
    # V rows: B12 is rows 0,1; cols 3,4,5 of B.
    # Local V row (j, k) -> Global V row j*7 + (3+k).
    v_rows_2 = [j * 7 + (3 + k) for j in range(2) for k in range(3)]
    # W rows: C12 is cols 3,4,5 of C; rows 0..2.
    # Local W row (k, i) -> Global W row (3+k)*3 + i.
    w_rows_2 = [(3 + k) * 3 + i for k in range(3) for i in range(3)]
    place_block(U2, V2, W2, u_rows_2, v_rows_2, w_rows_2, col_offset)
    col_offset += 18

    # Block 3: A1 (3x2) * B13 (2x1)
    # U rows: Same.
    u_rows_3 = [i * 5 + j for i in range(3) for j in range(2)]
    # V rows: B13 is rows 0,1; col 6 of B.
    # Local V row (j, k) -> Global V row j*7 + 6.
    v_rows_3 = [j * 7 + 6 for j in range(2) for k in range(1)]
    # W rows: C13 is col 6 of C; rows 0..2.
    # Local W row (k, i) -> Global W row 6*3 + i.
    w_rows_3 = [6 * 3 + i for k in range(1) for i in range(3)]
    place_block(U3, V3, W3, u_rows_3, v_rows_3, w_rows_3, col_offset)
    col_offset += 6

    # Block 4: A2 (3x3) * B21 (3x3)
    # U rows: A2 is cols 2,3,4 of A. i=0..2.
    # Local U row (i, j) -> Global U row i*5 + (2+j).
    u_rows_4 = [i * 5 + (2 + j) for i in range(3) for j in range(3)]
    # V rows: B21 is rows 2,3,4; cols 0,1,2 of B.
    # Local V row (j, k) -> Global V row (2+j)*7 + k.
    v_rows_4 = [(2 + j) * 7 + k for j in range(3) for k in range(3)]
    # W rows: C21 is cols 0,1,2 of C; rows 0..2.
    # Local W row (k, i) -> Global W row k*3 + i.
    w_rows_4 = [k * 3 + i for k in range(3) for i in range(3)]
    place_block(U4, V4, W4, u_rows_4, v_rows_4, w_rows_4, col_offset)
    col_offset += 9

    # Block 5: A2 (3x3) * B22 (3x3)
    # U rows: Same as Block 4.
    u_rows_5 = [i * 5 + (2 + j) for i in range(3) for j in range(3)]
    # V rows: B22 is rows 2,3,4; cols 3,4,5 of B.
    # Local V row (j, k) -> Global V row (2+j)*7 + (3+k).
    v_rows_5 = [(2 + j) * 7 + (3 + k) for j in range(3) for k in range(3)]
    # W rows: C22 is cols 3,4,5 of C; rows 0..2.
    # Local W row (k, i) -> Global W row (3+k)*3 + i.
    w_rows_5 = [(3 + k) * 3 + i for k in range(3) for i in range(3)]
    place_block(U5, V5, W5, u_rows_5, v_rows_5, w_rows_5, col_offset)
    col_offset += 9

    # Block 6: A2 (3x3) * B23 (3x1)
    # U rows: Same as Block 4.
    u_rows_6 = [i * 5 + (2 + j) for i in range(3) for j in range(3)]
    # V rows: B23 is rows 2,3,4; col 6 of B.
    # Local V row (j, k) -> Global V row (2+j)*7 + 6.
    v_rows_6 = [(2 + j) * 7 + 6 for j in range(3) for k in range(1)]
    # W rows: C23 is col 6 of C; rows 0..2.
    # Local W row (k, i) -> Global W row 6*3 + i.
    w_rows_6 = [6 * 3 + i for k in range(1) for i in range(3)]
    place_block(U6, V6, W6, u_rows_6, v_rows_6, w_rows_6, col_offset)
    col_offset += 9

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen block
    # x_idx: [x00, x01, x10, x11] indices in U
    # y_idx: [y00, y01, y10, y11] indices in V
    # w_offset: offset for W rows (0 for C_L, 4 for C_R)
    # base_col: starting column index
    def add_strassen_block(x_idx, y_idx, w_offset, base_col):
        x00, x01, x10, x11 = x_idx
        y00, y01, y10, y11 = y_idx
        
        # Col base_col: M_S1 = (x00+x11)(y00+y11)
        c = base_col
        U[x00][c] += 1
        U[x11][c] += 1
        V[y00][c] += 1
        V[y11][c] += 1
        W[0 + w_offset][c] += 1  # z00
        W[3 + w_offset][c] += 1  # z11
        
        # Col base_col+1: M_S2 = (x10+x11)y00
        c = base_col + 1
        U[x10][c] += 1
        U[x11][c] += 1
        V[y00][c] += 1
        W[1 + w_offset][c] += 1  # z10
        
        # Col base_col+2: M_S3 = x00(y01-y11)
        c = base_col + 2
        U[x00][c] += 1
        V[y01][c] += 1
        V[y11][c] -= 1
        W[2 + w_offset][c] += 1  # z01
        W[3 + w_offset][c] -= 1  # z11
        
        # Col base_col+3: M_S4 = x11(y10-y00)
        c = base_col + 3
        U[x11][c] += 1
        V[y10][c] += 1
        V[y00][c] -= 1
        W[0 + w_offset][c] -= 1  # z00
        W[1 + w_offset][c] += 1  # z10
        
        # Col base_col+4: M_S5 = (x00+x01)y11
        c = base_col + 4
        U[x00][c] += 1
        U[x01][c] += 1
        V[y11][c] += 1
        W[2 + w_offset][c] += 1  # z01
        
        # Col base_col+5: M_S6 = (x10-x00)(y00+y01)
        c = base_col + 5
        U[x10][c] += 1
        U[x00][c] -= 1
        V[y00][c] += 1
        V[y01][c] += 1
        W[0 + w_offset][c] += 1  # z00
        
        # Col base_col+6: M_S7 = (x01-x11)(y10+y11)
        c = base_col + 6
        U[x01][c] += 1
        U[x11][c] -= 1
        V[y10][c] += 1
        V[y11][c] += 1
        W[3 + w_offset][c] += 1  # z11

    # Block 1: A_L B_TL -> C_L
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0, 1, 4, 5
    # B_TL: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0, 1, 4, 5
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], 0, 0)

    # Block 2: A_R B_BL -> C_L
    # A_R: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 2, 3, 6, 7
    # B_BL: B[2,0], B[2,1], B[3,0], B[3,1] -> V indices 8, 9, 12, 13
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], 0, 7)

    # Block 3: A_L B_TR -> C_R
    # A_L: same as Block 1
    # B_TR: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices 2, 3, 6, 7
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], 4, 14)

    # Block 4: A_R B_BR -> C_R
    # A_R: same as Block 2
    # B_BR: B[2,2], B[2,3], B[3,2], B[3,3] -> V indices 10, 11, 14, 15
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], 4, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

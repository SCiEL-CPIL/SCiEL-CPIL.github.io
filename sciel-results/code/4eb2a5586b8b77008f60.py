"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 25 decomposition
    # 3 Strassen blocks (rank 7 each) for cols (0,1), (2,3), (4,5)
    # 1 Schoolbook block (rank 4) for col 6
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col_start, rank_offset):
        # B indices for this block
        # j=0: col_start, col_start+1
        # j=1: p+col_start, p+col_start+1
        b_idx_11 = 0 * p + col_start
        b_idx_12 = 0 * p + col_start + 1
        b_idx_21 = 1 * p + col_start
        b_idx_22 = 1 * p + col_start + 1
        
        # C indices for this block
        # i=0: col_start, col_start+1
        # i=1: n+col_start, n+col_start+1
        c_idx_11 = col_start * n + 0
        c_idx_12 = (col_start + 1) * n + 0
        c_idx_21 = col_start * n + 1
        c_idx_22 = (col_start + 1) * n + 1
        
        # A indices are always 0,1,2,3
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # Strassen factors
        # M1 = (a11+a22)(b11+b22)
        U[0][rank_offset] = 1
        U[3][rank_offset] = 1
        V[b_idx_11][rank_offset] = 1
        V[b_idx_22][rank_offset] = 1
        W[c_idx_11][rank_offset] = 1
        W[c_idx_22][rank_offset] = 1
        
        # M2 = (a21+a22)b11
        U[2][rank_offset+1] = 1
        U[3][rank_offset+1] = 1
        V[b_idx_11][rank_offset+1] = 1
        W[c_idx_21][rank_offset+1] = 1
        W[c_idx_22][rank_offset+1] = -1
        
        # M3 = a11(b12-b22)
        U[0][rank_offset+2] = 1
        V[b_idx_12][rank_offset+2] = 1
        V[b_idx_22][rank_offset+2] = -1
        W[c_idx_12][rank_offset+2] = 1
        W[c_idx_22][rank_offset+2] = 1
        
        # M4 = a22(b21-b11)
        U[3][rank_offset+3] = 1
        V[b_idx_21][rank_offset+3] = 1
        V[b_idx_11][rank_offset+3] = -1
        W[c_idx_11][rank_offset+3] = 1
        W[c_idx_21][rank_offset+3] = 1
        
        # M5 = (a11+a12)b22
        U[0][rank_offset+4] = 1
        U[1][rank_offset+4] = 1
        V[b_idx_22][rank_offset+4] = 1
        W[c_idx_11][rank_offset+4] = -1
        W[c_idx_12][rank_offset+4] = 1
        
        # M6 = (a21-a11)(b11+b12)
        U[2][rank_offset+5] = 1
        U[0][rank_offset+5] = -1
        V[b_idx_11][rank_offset+5] = 1
        V[b_idx_12][rank_offset+5] = 1
        W[c_idx_22][rank_offset+5] = 1
        
        # M7 = (a12-a22)(b21+b22)
        U[1][rank_offset+6] = 1
        U[3][rank_offset+6] = -1
        V[b_idx_21][rank_offset+6] = 1
        V[b_idx_22][rank_offset+6] = 1
        W[c_idx_11][rank_offset+6] = 1

    def add_schoolbook_block(col_start, rank_offset):
        # 2x2 x 1 multiplication
        # C_1k = a11 b1k + a12 b2k
        # C_2k = a21 b1k + a22 b2k
        
        b_idx_1k = 0 * p + col_start
        b_idx_2k = 1 * p + col_start
        c_idx_1k = col_start * n + 0
        c_idx_2k = col_start * n + 1
        
        # Term 1: a11 b1k -> C1k
        U[0][rank_offset] = 1
        V[b_idx_1k][rank_offset] = 1
        W[c_idx_1k][rank_offset] = 1
        
        # Term 2: a12 b2k -> C1k
        U[1][rank_offset+1] = 1
        V[b_idx_2k][rank_offset+1] = 1
        W[c_idx_1k][rank_offset+1] = 1
        
        # Term 3: a21 b1k -> C2k
        U[2][rank_offset+2] = 1
        V[b_idx_1k][rank_offset+2] = 1
        W[c_idx_2k][rank_offset+2] = 1
        
        # Term 4: a22 b2k -> C2k
        U[3][rank_offset+3] = 1
        V[b_idx_2k][rank_offset+3] = 1
        W[c_idx_2k][rank_offset+3] = 1

    add_strassen_block(0, 0)
    add_strassen_block(2, 7)
    add_strassen_block(4, 14)
    add_schoolbook_block(6, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

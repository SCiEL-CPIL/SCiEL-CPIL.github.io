"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Rank 56 using Strassen on 2x2 blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition
    # A is 2x2, B is 2x2
    # a, b, c, d are A entries
    # e, f, g, h are B entries
    # C is 2x2 result
    
    # Block indices
    # A has 2 blocks horizontally: 0 (cols 0-1), 1 (cols 2-3)
    # B has 2 blocks vertically: 0 (rows 0-1), 1 (rows 2-3)
    # B has 4 blocks horizontally: 0 (cols 0-1), 1 (cols 2-3), 2 (cols 4-5), 3 (cols 6-7)
    
    col = 0
    for a_block in range(2): # 0: A_L, 1: A_R
        # A row offset is always 0
        a_col_off = a_block * 2
        
        # If a_block is 0, B uses top rows (0-1)
        # If a_block is 1, B uses bottom rows (2-3)
        b_row_off = a_block * 2
        
        for b_col_block in range(4):
            b_col_off = b_col_block * 2
            
            # Indices in U (n*m)
            # a: i=0, j=a_col_off
            # b: i=0, j=a_col_off+1
            # c: i=1, j=a_col_off
            # d: i=1, j=a_col_off+1
            idx_a = 0 * m + a_col_off
            idx_b = 0 * m + a_col_off + 1
            idx_c = 1 * m + a_col_off
            idx_d = 1 * m + a_col_off + 1
            
            # Indices in V (m*p)
            # e: j=b_row_off, k=b_col_off
            # f: j=b_row_off, k=b_col_off+1
            # g: j=b_row_off+1, k=b_col_off
            # h: j=b_row_off+1, k=b_col_off+1
            idx_e = b_row_off * p + b_col_off
            idx_f = b_row_off * p + b_col_off + 1
            idx_g = (b_row_off + 1) * p + b_col_off
            idx_h = (b_row_off + 1) * p + b_col_off + 1
            
            # Indices in W (p*n)
            # c00: k=b_col_off, i=0 -> k*n + i
            # c01: k=b_col_off+1, i=0
            # c10: k=b_col_off, i=1
            # c11: k=b_col_off+1, i=1
            idx_c00 = b_col_off * n + 0
            idx_c01 = (b_col_off + 1) * n + 0
            idx_c10 = b_col_off * n + 1
            idx_c11 = (b_col_off + 1) * n + 1
            
            # M1 = (a+d)(e+h)
            U[idx_a][col] += 1
            U[idx_d][col] += 1
            V[idx_e][col] += 1
            V[idx_h][col] += 1
            W[idx_c00][col] += 1
            W[idx_c11][col] += 1
            col += 1
            
            # M2 = (c+d)e
            U[idx_c][col] += 1
            U[idx_d][col] += 1
            V[idx_e][col] += 1
            W[idx_c10][col] += 1
            col += 1
            
            # M3 = a(f-h)
            U[idx_a][col] += 1
            V[idx_f][col] += 1
            V[idx_h][col] += -1
            W[idx_c01][col] += 1
            W[idx_c11][col] += -1
            col += 1
            
            # M4 = d(g-e)
            U[idx_d][col] += 1
            V[idx_g][col] += 1
            V[idx_e][col] += -1
            W[idx_c00][col] += -1
            W[idx_c10][col] += 1
            col += 1
            
            # M5 = (a+b)h
            U[idx_a][col] += 1
            U[idx_b][col] += 1
            V[idx_h][col] += 1
            W[idx_c00][col] += 1
            W[idx_c01][col] += 1
            col += 1
            
            # M6 = (c-a)(e+f)
            U[idx_c][col] += 1
            U[idx_a][col] += -1
            V[idx_e][col] += 1
            V[idx_f][col] += 1
            W[idx_c11][col] += 1
            col += 1
            
            # M7 = (b-d)(g+h)
            U[idx_b][col] += 1
            U[idx_d][col] += -1
            V[idx_g][col] += 1
            V[idx_h][col] += 1
            W[idx_c00][col] += -1
            W[idx_c11][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

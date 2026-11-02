import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    # U: (20, rank)
    # V: (30, rank)
    # W: (24, rank)
    
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2 block
    def add_strassen_2x2(rA0, cA0, rB0, cB0, col_start):
        # A block at (rA0, cA0) size 2x2
        # B block at (rB0, cB0) size 2x2
        # C block at (rA0, cB0) size 2x2
        
        u_a = rA0 * m + cA0
        u_b = rA0 * m + (cA0 + 1)
        u_c = (rA0 + 1) * m + cA0
        u_d = (rA0 + 1) * m + (cA0 + 1)
        
        v_e = rB0 * p + cB0
        v_f = rB0 * p + (cB0 + 1)
        v_g = (rB0 + 1) * p + cB0
        v_h = (rB0 + 1) * p + (cB0 + 1)
        
        w_00 = cB0 * n + rA0
        w_01 = (cB0 + 1) * n + rA0
        w_10 = cB0 * n + (rA0 + 1)
        w_11 = (cB0 + 1) * n + (rA0 + 1)
        
        # M1: (a+d)(e+h)
        U[u_a][col_start] = 1
        U[u_d][col_start] = 1
        V[v_e][col_start] = 1
        V[v_h][col_start] = 1
        W[w_00][col_start] = 1
        W[w_01][col_start] = 1
        W[w_10][col_start] = 1
        W[w_11][col_start] = 1
        
        # M2: (c+d)e
        U[u_c][col_start+1] = 1
        U[u_d][col_start+1] = 1
        V[v_e][col_start+1] = 1
        W[w_10][col_start+1] = 1
        W[w_11][col_start+1] = 1
        
        # M3: a(f-h)
        U[u_a][col_start+2] = 1
        V[v_f][col_start+2] = 1
        V[v_h][col_start+2] = -1
        W[w_01][col_start+2] = 1
        W[w_11][col_start+2] = 1
        
        # M4: d(g-e)
        U[u_d][col_start+3] = 1
        V[v_g][col_start+3] = 1
        V[v_e][col_start+3] = -1
        W[w_00][col_start+3] = 1
        W[w_11][col_start+3] = 1
        
        # M5: (a+b)h
        U[u_a][col_start+4] = 1
        U[u_b][col_start+4] = 1
        V[v_h][col_start+4] = 1
        W[w_00][col_start+4] = -1
        W[w_10][col_start+4] = 1
        
        # M6: (c-a)(e+f)
        U[u_c][col_start+5] = 1
        U[u_a][col_start+5] = -1
        V[v_e][col_start+5] = 1
        V[v_f][col_start+5] = 1
        W[w_10][col_start+5] = 1
        W[w_11][col_start+5] = -1
        
        # M7: (b-d)(g+h)
        U[u_b][col_start+6] = 1
        U[u_d][col_start+6] = -1
        V[v_g][col_start+6] = 1
        V[v_h][col_start+6] = 1
        W[w_00][col_start+6] = 1
        W[w_01][col_start+6] = -1
        
        return col_start + 7

    # Term 1: A1 (4x2) * B1 (2x6)
    # A1 is cols 0,1 of A.
    # B1 is rows 0,1 of B.
    # Split A1 into top (rows 0,1) and bot (rows 2,3).
    # Split B1 into three 2x2 blocks: cols 0-1, 2-3, 4-5.
    
    # A1_top * B1_1
    col = add_strassen_2x2(0, 0, 0, 0, col)
    # A1_top * B1_2
    col = add_strassen_2x2(0, 0, 0, 2, col)
    # A1_top * B1_3
    col = add_strassen_2x2(0, 0, 0, 4, col)
    
    # A1_bot * B1_1
    col = add_strassen_2x2(2, 0, 0, 0, col)
    # A1_bot * B1_2
    col = add_strassen_2x2(2, 0, 0, 2, col)
    # A1_bot * B1_3
    col = add_strassen_2x2(2, 0, 0, 4, col)
    
    # Term 2: A2 (4x2) * B2 (2x6)
    # A2 is cols 2,3 of A.
    # B2 is rows 2,3 of B.
    
    # A2_top * B2_1
    col = add_strassen_2x2(0, 2, 2, 0, col)
    # A2_top * B2_2
    col = add_strassen_2x2(0, 2, 2, 2, col)
    # A2_top * B2_3
    col = add_strassen_2x2(0, 2, 2, 4, col)
    
    # A2_bot * B2_1
    col = add_strassen_2x2(2, 2, 2, 0, col)
    # A2_bot * B2_2
    col = add_strassen_2x2(2, 2, 2, 2, col)
    # A2_bot * B2_3
    col = add_strassen_2x2(2, 2, 2, 4, col)
    
    # Term 3: A3 (4x1) * B3 (1x6)
    # A3 is col 4 of A.
    # B3 is row 4 of B.
    
    # U[i, col] = A3[i] -> 1
    # V[4*6 + j, col] = B3[j] -> 1
    # W[j*4 + i, col] = 1
    
    for i in range(4):
        U[i * m + 4][col] = 1
    for j in range(6):
        V[4 * p + j][col] = 1
    for i in range(4):
        for j in range(6):
            W[j * n + i][col] = 1
            
    col += 1
    
    return {"U": U, "V": V, "W": W}

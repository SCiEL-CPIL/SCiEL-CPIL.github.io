"""Matrix multiplication 3x4x7 - Strassen-based decomposition."""
import time

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Construct a rank-78 tensor decomposition for 3x4x7 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    rank = 78
    U = [[0] * rank for _ in range(n * m)]  # 12 x 78
    V = [[0] * rank for _ in range(m * p)]  # 28 x 78
    W = [[0] * rank for _ in range(p * n)]  # 21 x 78
    
    col = 0
    
    def add_strassen_2x2x2(u_rows, v_rows, w_rows, col):
        """
        u_rows: [U(0,0), U(0,1), U(1,0), U(1,1)]
        v_rows: [V(0,0), V(0,1), V(1,0), V(1,1)]
        w_rows: [W(0,0), W(0,1), W(1,0), W(1,1)]  where W(k,i)
        """
        # M1: (a00+a11)*(b00+b11)
        U[u_rows[0]][col] += 1
        U[u_rows[3]][col] += 1
        V[v_rows[0]][col] += 1
        V[v_rows[3]][col] += 1
        W[w_rows[0]][col] += 1
        W[w_rows[3]][col] += 1
        col += 1
        
        # M2: (a10+a11)*b00
        U[u_rows[2]][col] += 1
        U[u_rows[3]][col] += 1
        V[v_rows[0]][col] += 1
        W[w_rows[1]][col] += 1
        col += 1
        
        # M3: a00*(b01-b11)
        U[u_rows[0]][col] += 1
        V[v_rows[1]][col] += 1
        V[v_rows[3]][col] -= 1
        W[w_rows[0]][col] += 1
        col += 1
        
        # M4: a11*(b10-b00)
        U[u_rows[3]][col] += 1
        V[v_rows[2]][col] += 1
        V[v_rows[0]][col] -= 1
        W[w_rows[1]][col] += 1
        col += 1
        
        # M5: (a00+a01)*b11
        U[u_rows[0]][col] += 1
        U[u_rows[1]][col] += 1
        V[v_rows[3]][col] += 1
        W[w_rows[2]][col] += 1
        col += 1
        
        # M6: (a10-a00)*(b00+b01)
        U[u_rows[2]][col] += 1
        U[u_rows[0]][col] -= 1
        V[v_rows[0]][col] += 1
        V[v_rows[1]][col] += 1
        W[w_rows[0]][col] += 1
        col += 1
        
        # M7: (a01-a11)*(b10+b11)
        U[u_rows[1]][col] += 1
        U[u_rows[3]][col] -= 1
        V[v_rows[2]][col] += 1
        V[v_rows[3]][col] += 1
        W[w_rows[3]][col] += 1
        col += 1
        
        return col
    
    def add_schoolbook_2x2x1(u_rows, v_rows, w_rows, col):
        """
        u_rows: [U(0,0), U(0,1), U(1,0), U(1,1)]
        v_rows: [V(0,0), V(1,0)]
        w_rows: [W(0,0), W(0,1)]  where W(k,i), k=0
        """
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                U[u_rows[idx]][col] = 1
                V[v_rows[j]][col] = 1
                W[w_rows[i]][col] = 1
                col += 1
        return col
    
    def add_schoolbook_1x2x2(u_rows, v_rows, w_rows, col):
        """
        u_rows: [U(0,0), U(0,1)]
        v_rows: [V(0,0), V(0,1), V(1,0), V(1,1)]
        w_rows: [W(0,0), W(1,0)]  where W(k,i), i=0
        """
        for j in range(2):
            for k in range(2):
                v_idx = j * 2 + k
                U[u_rows[j]][col] = 1
                V[v_rows[v_idx]][col] = 1
                W[w_rows[k]][col] = 1
                col += 1
        return col
    
    def add_schoolbook_1x2x1(u_rows, v_rows, w_rows, col):
        """
        u_rows: [U(0,0), U(0,1)]
        v_rows: [V(0,0), V(1,0)]
        w_rows: [W(0,0)]  where W(k,i), k=0, i=0
        """
        for j in range(2):
            U[u_rows[j]][col] = 1
            V[v_rows[j]][col] = 1
            W[w_rows[0]][col] = 1
            col += 1
        return col
    
    # Part 1: A1 (rows 0,1) x B (4x7)
    # A1a (cols 0,1) x B_top (rows 0,1)
    u_rows_a1a = [0, 1, 4, 5]  # i*4+j
    
    # B_top cols 0,1
    v_rows_btop_01 = [0, 1, 7, 8]  # j*7+k
    w_rows_c01 = [0, 1, 3, 4]  # k*3+i
    col = add_strassen_2x2x2(u_rows_a1a, v_rows_btop_01, w_rows_c01, col)
    
    # B_top cols 2,3
    v_rows_btop_23 = [2, 3, 9, 10]
    col = add_strassen_2x2x2(u_rows_a1a, v_rows_btop_23, w_rows_c01, col)
    
    # B_top cols 4,5
    v_rows_btop_45 = [4, 5, 11, 12]
    col = add_strassen_2x2x2(u_rows_a1a, v_rows_btop_45, w_rows_c01, col)
    
    # B_top col 6
    v_rows_btop_6 = [6, 13]
    w_rows_c0 = [0, 1]
    col = add_schoolbook_2x2x1(u_rows_a1a, v_rows_btop_6, w_rows_c0, col)
    
    # A1b (cols 2,3) x B_bottom (rows 2,3)
    u_rows_a1b = [2, 3, 6, 7]
    
    # B_bottom cols 0,1
    v_rows_bbot_01 = [14, 15, 21, 22]
    col = add_strassen_2x2x2(u_rows_a1b, v_rows_bbot_01, w_rows_c01, col)
    
    # B_bottom cols 2,3
    v_rows_bbot_23 = [16, 17, 23, 24]
    col = add_strassen_2x2x2(u_rows_a1b, v_rows_bbot_23, w_rows_c01, col)
    
    # B_bottom cols 4,5
    v_rows_bbot_45 = [18, 19, 25, 26]
    col = add_strassen_2x2x2(u_rows_a1b, v_rows_bbot_45, w_rows_c01, col)
    
    # B_bottom col 6
    v_rows_bbot_6 = [20, 27]
    col = add_schoolbook_2x2x1(u_rows_a1b, v_rows_bbot_6, w_rows_c0, col)
    
    # Part 2: A2 (row 2) x B
    # A2a (cols 0,1) x B_top (rows 0,1)
    u_rows_a2a = [8, 9]
    
    # B_top cols 0,1
    w_rows_c2 = [2, 5]  # k*3+2 for k in {0,1}
    col = add_schoolbook_1x2x2(u_rows_a2a, v_rows_btop_01, w_rows_c2, col)
    
    # B_top cols 2,3
    col = add_schoolbook_1x2x2(u_rows_a2a, v_rows_btop_23, w_rows_c2, col)
    
    # B_top cols 4,5
    col = add_schoolbook_1x2x2(u_rows_a2a, v_rows_btop_45, w_rows_c2, col)
    
    # B_top col 6
    w_rows_c2_0 = [2]
    col = add_schoolbook_1x2x1(u_rows_a2a, v_rows_btop_6, w_rows_c2_0, col)
    
    # A2b (cols 2,3) x B_bottom (rows 2,3)
    u_rows_a2b = [10, 11]
    
    # B_bottom cols 0,1
    col = add_schoolbook_1x2x2(u_rows_a2b, v_rows_bbot_01, w_rows_c2, col)
    
    # B_bottom cols 2,3
    col = add_schoolbook_1x2x2(u_rows_a2b, v_rows_bbot_23, w_rows_c2, col)
    
    # B_bottom cols 4,5
    col = add_schoolbook_1x2x2(u_rows_a2b, v_rows_bbot_45, w_rows_c2, col)
    
    # B_bottom col 6
    col = add_schoolbook_1x2x1(u_rows_a2b, v_rows_bbot_6, w_rows_c2_0, col)
    
    return {"U": U, "V": V, "W": W}

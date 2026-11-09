import itertools

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the rank-54 decomposition
    # 6 Strassen blocks (rank 7 each) + 1 Outer product block (rank 12)
    rank = 54
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add Strassen decomposition for 2x2 * 2x2
    # A block rows: ai0, ai1; cols: aj0, aj1
    # B block rows: bj0, bj1; cols: bk0, bk1
    def add_strassen(ai0, aj0, bj0, bk0):
        nonlocal col_idx
        # A indices in U
        # A[ai0, aj0] -> U[ai0*m + aj0]
        # A[ai0, aj1] -> U[ai0*m + aj0+1]
        # A[ai1, aj0] -> U[ai1*m + aj0]
        # A[ai1, aj1] -> U[ai1*m + aj0+1]
        
        # B indices in V
        # B[bj0, bk0] -> V[bj0*p + bk0]
        # B[bj0, bk1] -> V[bj0*p + bk0+1]
        # B[bj1, bk0] -> V[bj1*p + bk0]
        # B[bj1, bk1] -> V[bj1*p + bk0+1]
        
        # C indices in W
        # C[ai0, bk0] -> W[bk0*n + ai0]
        # C[ai0, bk1] -> W[bk1*n + ai0]
        # C[ai1, bk0] -> W[bk0*n + ai1]
        # C[ai1, bk1] -> W[bk1*n + ai1]
        
        u_a = ai0 * m + aj0
        u_b = ai0 * m + aj0 + 1
        u_c = ai1 * m + aj0
        u_d = ai1 * m + aj0 + 1
        
        v_e = bj0 * p + bk0
        v_f = bj0 * p + bk0 + 1
        v_g = bj1 * p + bk0
        v_h = bj1 * p + bk0 + 1
        
        w_c11 = bk0 * n + ai0
        w_c12 = bk1 * n + ai0
        w_c21 = bk0 * n + ai1
        w_c22 = bk1 * n + ai1
        
        # P1 = (a+d)(e+h)
        U[u_a][col_idx] = 1.0
        U[u_d][col_idx] = 1.0
        V[v_e][col_idx] = 1.0
        V[v_h][col_idx] = 1.0
        W[w_c11][col_idx] = 1.0
        W[w_c12][col_idx] = 0.0
        W[w_c21][col_idx] = 0.0
        W[w_c22][col_idx] = 1.0
        col_idx += 1
        
        # P2 = (c+d)e
        U[u_c][col_idx] = 1.0
        U[u_d][col_idx] = 1.0
        V[v_e][col_idx] = 1.0
        W[w_c11][col_idx] = 0.0
        W[w_c12][col_idx] = 0.0
        W[w_c21][col_idx] = 1.0
        W[w_c22][col_idx] = -1.0
        col_idx += 1
        
        # P3 = a(f-h)
        U[u_a][col_idx] = 1.0
        V[v_f][col_idx] = 1.0
        V[v_h][col_idx] = -1.0
        W[w_c11][col_idx] = 0.0
        W[w_c12][col_idx] = 1.0
        W[w_c21][col_idx] = 0.0
        W[w_c22][col_idx] = 1.0
        col_idx += 1
        
        # P4 = d(g-e)
        U[u_d][col_idx] = 1.0
        V[v_g][col_idx] = 1.0
        V[v_e][col_idx] = -1.0
        W[w_c11][col_idx] = 1.0
        W[w_c12][col_idx] = 0.0
        W[w_c21][col_idx] = 1.0
        W[w_c22][col_idx] = 0.0
        col_idx += 1
        
        # P5 = (a+b)h
        U[u_a][col_idx] = 1.0
        U[u_b][col_idx] = 1.0
        V[v_h][col_idx] = 1.0
        W[w_c11][col_idx] = -1.0
        W[w_c12][col_idx] = 1.0
        W[w_c21][col_idx] = 0.0
        W[w_c22][col_idx] = 0.0
        col_idx += 1
        
        # P6 = (c-a)(e+f)
        U[u_c][col_idx] = 1.0
        U[u_a][col_idx] = -1.0
        V[v_e][col_idx] = 1.0
        V[v_f][col_idx] = 1.0
        W[w_c11][col_idx] = 0.0
        W[w_c12][col_idx] = 0.0
        W[w_c21][col_idx] = 0.0
        W[w_c22][col_idx] = 1.0
        col_idx += 1
        
        # P7 = (b-d)(g+h)
        U[u_b][col_idx] = 1.0
        U[u_d][col_idx] = -1.0
        V[v_g][col_idx] = 1.0
        V[v_h][col_idx] = 1.0
        W[w_c11][col_idx] = 1.0
        W[w_c12][col_idx] = 0.0
        W[w_c21][col_idx] = 0.0
        W[w_c22][col_idx] = 0.0
        col_idx += 1

    # Helper to add outer product decomposition for 2x1 * 1x6
    # A block row: ai0, ai1; col: aj0
    # B block row: bj0; cols: bk0..bk0+5
    def add_outer(ai0, aj0, bj0, bk0):
        nonlocal col_idx
        for i in [0, 1]:
            for k in range(6):
                u_row = (ai0 + i) * m + aj0
                v_row = bj0 * p + (bk0 + k)
                w_row = (bk0 + k) * n + (ai0 + i)
                
                U[u_row][col_idx] = 1.0
                V[v_row][col_idx] = 1.0
                W[w_row][col_idx] = 1.0
                col_idx += 1

    # Block 1: A cols 0,1 (rows 0,1) with B rows 0,1
    # B cols 0,1
    add_strassen(0, 0, 0, 0)
    # B cols 2,3
    add_strassen(0, 0, 0, 2)
    # B cols 4,5
    add_strassen(0, 0, 0, 4)
    
    # Block 2: A cols 2,3 (rows 0,1) with B rows 2,3
    # B cols 0,1
    add_strassen(0, 2, 2, 0)
    # B cols 2,3
    add_strassen(0, 2, 2, 2)
    # B cols 4,5
    add_strassen(0, 2, 2, 4)
    
    # Block 3: A col 4 (rows 0,1) with B row 4
    # B cols 0..5
    add_outer(0, 4, 4, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 23 decomposition
    # 3 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x1 (Schoolbook, rank 2)
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    def add_strassen_block(b_start_col, u_offset, v_offset, w_offset):
        # b_start_col is the starting column index in B (0-indexed)
        # A is 2x2: indices 0,1,2,3 in U (i*m+j)
        # B block is 2x2: indices (j*p + k) in V
        # C block is 2x2: indices (k*n + i) in W
        
        # Strassen products:
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        
        # U indices:
        # a: 0, b: 1, c: 2, d: 3
        
        # V indices for block:
        # e: j=0, k=0 -> 0*p + b_start_col
        # f: j=0, k=1 -> 0*p + b_start_col + 1
        # g: j=1, k=0 -> 1*p + b_start_col
        # h: j=1, k=1 -> 1*p + b_start_col + 1
        
        v_e = 0 * p + b_start_col
        v_f = 0 * p + b_start_col + 1
        v_g = 1 * p + b_start_col
        v_h = 1 * p + b_start_col + 1
        
        # W indices for block:
        # C = [[c00, c01], [c10, c11]]
        # W index for c_ij is k*n + i
        # c00: k=0, i=0 -> 0
        # c01: k=1, i=0 -> 1
        # c10: k=0, i=1 -> 2
        # c11: k=1, i=1 -> 3
        # But these are relative to the block.
        # Global W index: (k_global * n + i)
        # k_global for block is b_start_col + k_local
        
        w_c00 = (b_start_col + 0) * n + 0
        w_c01 = (b_start_col + 1) * n + 0
        w_c10 = (b_start_col + 0) * n + 1
        w_c11 = (b_start_col + 1) * n + 1
        
        # P1
        U[0][u_offset] += 1
        U[3][u_offset] += 1
        V[v_e][u_offset] += 1
        V[v_h][u_offset] += 1
        W[w_c00][u_offset] += 1
        W[w_c11][u_offset] += 1
        u_offset += 1
        
        # P2
        U[2][u_offset] += 1
        U[3][u_offset] += 1
        V[v_e][u_offset] += 1
        W[w_c10][u_offset] += 1
        W[w_c11][u_offset] -= 1
        u_offset += 1
        
        # P3
        U[0][u_offset] += 1
        V[v_f][u_offset] += 1
        V[v_h][u_offset] -= 1
        W[w_c01][u_offset] += 1
        W[w_c11][u_offset] += 1
        u_offset += 1
        
        # P4
        U[3][u_offset] += 1
        V[v_g][u_offset] += 1
        V[v_e][u_offset] -= 1
        W[w_c00][u_offset] += 1
        W[w_c10][u_offset] += 1
        u_offset += 1
        
        # P5
        U[0][u_offset] += 1
        U[1][u_offset] += 1
        V[v_h][u_offset] += 1
        W[w_c00][u_offset] -= 1
        W[w_c01][u_offset] += 1
        u_offset += 1
        
        # P6
        U[2][u_offset] += 1
        U[0][u_offset] -= 1
        V[v_e][u_offset] += 1
        V[v_f][u_offset] += 1
        W[w_c10][u_offset] += 1
        W[w_c11][u_offset] += 1
        u_offset += 1
        
        # P7
        U[1][u_offset] += 1
        U[3][u_offset] -= 1
        V[v_g][u_offset] += 1
        V[v_h][u_offset] += 1
        W[w_c00][u_offset] += 1
        W[w_c01][u_offset] -= 1
        u_offset += 1
        
        return u_offset, v_offset, w_offset

    # Block 0: cols 0,1
    col_offset = 0
    u_off, v_off, w_off = add_strassen_block(0, 0, 0, 0)
    
    # Block 1: cols 2,3
    col_offset = 2
    u_off, v_off, w_off = add_strassen_block(2, u_off, v_off, w_off)
    
    # Block 2: cols 4,5
    col_offset = 4
    u_off, v_off, w_off = add_strassen_block(4, u_off, v_off, w_off)
    
    # Block 3: col 6 (2x1)
    # Schoolbook: 2 terms
    # C_0,6 = a*B_0,6 + b*B_1,6
    # C_1,6 = c*B_0,6 + d*B_1,6
    
    # Term 1: a * B_0,6
    U[0][u_off] = 1
    V[0 * p + 6][u_off] = 1
    W[6 * n + 0][u_off] = 1
    u_off += 1
    
    # Term 2: b * B_1,6
    U[1][u_off] = 1
    V[1 * p + 6][u_off] = 1
    W[6 * n + 0][u_off] = 1
    u_off += 1
    
    # Term 3: c * B_0,6
    U[2][u_off] = 1
    V[0 * p + 6][u_off] = 1
    W[6 * n + 1][u_off] = 1
    u_off += 1
    
    # Term 4: d * B_1,6
    U[3][u_off] = 1
    V[1 * p + 6][u_off] = 1
    W[6 * n + 1][u_off] = 1
    u_off += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

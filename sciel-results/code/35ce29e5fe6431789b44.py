"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 66 solution: 6 blocks of rank 11
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 3x2x2 decomposition
    def add_322(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # Local indices for A' (3x2), B' (2x2), C' (3x2)
        # A' rows 0,1,2; cols 0,1
        # B' rows 0,1; cols 0,1
        # C' rows 0,1,2; cols 0,1
        
        # Global indices
        def u_idx(r, c): return (a_row_offset + r) * m + (a_col_offset + c)
        def v_idx(r, c): return (b_row_offset + r) * p + (b_col_offset + c)
        def w_idx(r, c): return (c_col_offset + c) * n + (c_row_offset + r)
        
        # M1: (a00 + a11)(b00 + b11)
        U[u_idx(0, 0)][col] = 1
        U[u_idx(1, 1)][col] = 1
        V[v_idx(0, 0)][col] = 1
        V[v_idx(1, 1)][col] = 1
        W[w_idx(0, 0)][col] = 1
        W[w_idx(1, 1)][col] = 1
        # C00 += M1, C11 += M1
        col += 1
        
        # M2: (a10 + a11)b00
        U[u_idx(1, 0)][col] = 1
        U[u_idx(1, 1)][col] = 1
        V[v_idx(0, 0)][col] = 1
        W[w_idx(1, 0)][col] = 1
        W[w_idx(1, 1)][col] = -1
        # C10 += M2, C11 -= M2
        col += 1
        
        # M3: a00(b01 - b11)
        U[u_idx(0, 0)][col] = 1
        V[v_idx(0, 1)][col] = 1
        V[v_idx(1, 1)][col] = -1
        W[w_idx(0, 1)][col] = 1
        W[w_idx(1, 1)][col] = 1
        # C01 += M3, C11 += M3
        col += 1
        
        # M4: a11(b10 - b00)
        U[u_idx(1, 1)][col] = 1
        V[v_idx(1, 0)][col] = 1
        V[v_idx(0, 0)][col] = -1
        W[w_idx(0, 0)][col] = 1
        W[w_idx(1, 0)][col] = 1
        # C00 += M4, C10 += M4
        col += 1
        
        # M5: (a00 + a10)b11
        U[u_idx(0, 0)][col] = 1
        U[u_idx(1, 0)][col] = 1
        V[v_idx(1, 1)][col] = 1
        W[w_idx(0, 0)][col] = -1
        W[w_idx(0, 1)][col] = 1
        # C00 -= M5, C01 += M5
        col += 1
        
        # M6: (a01 - a00)(b00 + b10)
        U[u_idx(0, 1)][col] = 1
        U[u_idx(0, 0)][col] = -1
        V[v_idx(0, 0)][col] = 1
        V[v_idx(1, 0)][col] = 1
        W[w_idx(1, 1)][col] = 1
        # C11 += M6
        col += 1
        
        # M7: (a10 - a11)(b01 + b11)
        U[u_idx(1, 0)][col] = 1
        U[u_idx(1, 1)][col] = -1
        V[v_idx(0, 1)][col] = 1
        V[v_idx(1, 1)][col] = 1
        W[w_idx(0, 0)][col] = 1
        # C00 += M7
        col += 1
        
        # M8: a20 b00
        U[u_idx(2, 0)][col] = 1
        V[v_idx(0, 0)][col] = 1
        W[w_idx(2, 0)][col] = 1
        col += 1
        
        # M9: a21 b10
        U[u_idx(2, 1)][col] = 1
        V[v_idx(1, 0)][col] = 1
        W[w_idx(2, 0)][col] = 1
        col += 1
        
        # M10: a20 b01
        U[u_idx(2, 0)][col] = 1
        V[v_idx(0, 1)][col] = 1
        W[w_idx(2, 1)][col] = 1
        col += 1
        
        # M11: a21 b11
        U[u_idx(2, 1)][col] = 1
        V[v_idx(1, 1)][col] = 1
        W[w_idx(2, 1)][col] = 1
        col += 1

    # Loop over 2 blocks of m and 3 blocks of p
    for i in range(2):
        for k in range(3):
            # A offset: row 0, col 2*i
            # B offset: row 2*i, col 2*k
            # C offset: row 0, col 2*k
            add_322(0, 2*i, 2*i, 2*k, 0, 2*k)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

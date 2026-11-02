"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 terms per block
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # A = [[a00, a01], [a10, a11]] -> U indices 0, 1, 2, 3
    # B = [[b00, b01], [b10, b11]] -> V indices (local) 0, 1, 2, 3
    # C = [[c00, c01], [c10, c11]] -> W indices (local) 0, 1, 2, 3
    
    # Terms:
    # P1: (a00+a11)(b00+b11) -> C00 += P1, C11 += P1
    # P2: (a10+a11)b00 -> C10 += P2, C11 -= P2
    # P3: a00(b01-b11) -> C01 += P3, C11 += P3
    # P4: a11(b10-b00) -> C00 += P4, C10 += P4
    # P5: (a00+a01)b11 -> C00 -= P5, C01 += P5
    # P6: (a10-a00)(b00+b01) -> C11 += P6
    # P7: (a01-a11)(b10+b11) -> C00 += P7
    
    # Note on indices in the flattened tensors:
    # U index for a_ij is i*m + j
    # V index for b_jk is j*p + k
    # W index for c_ki is k*n + i
    
    for block in range(4):
        # Columns in B and C for this block
        col_offset = block * 2
        
        # V indices for b_00, b_01, b_10, b_11 of this block
        # b_00 is at row 0, col col_offset -> index 0*p + col_offset = col_offset
        # b_01 is at row 0, col col_offset+1 -> index 0*p + col_offset+1 = col_offset+1
        # b_10 is at row 1, col col_offset -> index 1*p + col_offset = 8 + col_offset
        # b_11 is at row 1, col col_offset+1 -> index 1*p + col_offset+1 = 8 + col_offset+1
        v_00 = col_offset
        v_01 = col_offset + 1
        v_10 = 8 + col_offset
        v_11 = 8 + col_offset + 1
        
        # W indices for c_00, c_01, c_10, c_11 of this block
        # c_00 is at row 0, col col_offset -> index col_offset*n + 0 = 2*col_offset
        # c_01 is at row 0, col col_offset+1 -> index (col_offset+1)*n + 0 = 2*(col_offset+1)
        # c_10 is at row 1, col col_offset -> index col_offset*n + 1 = 2*col_offset + 1
        # c_11 is at row 1, col col_offset+1 -> index (col_offset+1)*n + 1 = 2*(col_offset+1) + 1
        w_00 = 2 * col_offset
        w_01 = 2 * (col_offset + 1)
        w_10 = 2 * col_offset + 1
        w_11 = 2 * (col_offset + 1) + 1
        
        # U indices are always the same
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        # Start column index for this block's 7 terms
        base_col = block * 7
        
        # Term 1: P1
        col = base_col
        U[u_00][col] += 1
        U[u_11][col] += 1
        V[v_00][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        
        # Term 2: P2
        col = base_col + 1
        U[u_10][col] += 1
        U[u_11][col] += 1
        V[v_00][col] += 1
        W[w_10][col] += 1
        W[w_11][col] -= 1
        
        # Term 3: P3
        col = base_col + 2
        U[u_00][col] += 1
        V[v_01][col] += 1
        V[v_11][col] -= 1
        W[w_01][col] += 1
        W[w_11][col] += 1
        
        # Term 4: P4
        col = base_col + 3
        U[u_11][col] += 1
        V[v_10][col] += 1
        V[v_00][col] -= 1
        W[w_00][col] += 1
        W[w_10][col] += 1
        
        # Term 5: P5
        col = base_col + 4
        U[u_00][col] += 1
        U[u_01][col] += 1
        V[v_11][col] += 1
        W[w_00][col] -= 1
        W[w_01][col] += 1
        
        # Term 6: P6
        col = base_col + 5
        U[u_10][col] += 1
        U[u_00][col] -= 1
        V[v_00][col] += 1
        V[v_01][col] += 1
        W[w_11][col] += 1
        
        # Term 7: P7
        col = base_col + 6
        U[u_01][col] += 1
        U[u_11][col] -= 1
        V[v_10][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

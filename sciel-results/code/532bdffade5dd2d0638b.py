"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i, start_col):
        col = start_col
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[(u_off_i+0)*m + (u_off_j+0)][col] = 1
        U[(u_off_i+1)*m + (u_off_j+1)][col] = 1
        V[(v_off_j+0)*p + (v_off_k+0)][col] = 1
        V[(v_off_j+1)*p + (v_off_k+1)][col] = 1
        W[(w_off_k+0)*n + (w_off_i+0)][col] = 1
        W[(w_off_k+1)*n + (w_off_i+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10, C11
        U[(u_off_i+1)*m + (u_off_j+0)][col] = 1
        U[(u_off_i+1)*m + (u_off_j+1)][col] = 1
        V[(v_off_j+0)*p + (v_off_k+0)][col] = 1
        W[(w_off_k+0)*n + (w_off_i+1)][col] = 1
        W[(w_off_k+1)*n + (w_off_i+1)][col] = 1
        col += 1
        
        # M3: A00(B10-B11) -> C00, C01
        U[(u_off_i+0)*m + (u_off_j+0)][col] = 1
        V[(v_off_j+1)*p + (v_off_k+0)][col] = 1
        V[(v_off_j+1)*p + (v_off_k+1)][col] = -1
        W[(w_off_k+0)*n + (w_off_i+0)][col] = 1
        W[(w_off_k+1)*n + (w_off_i+0)][col] = 1
        col += 1
        
        # M4: A11(B00-B01) -> C00, C10
        U[(u_off_i+1)*m + (u_off_j+1)][col] = 1
        V[(v_off_j+0)*p + (v_off_k+0)][col] = 1
        V[(v_off_j+0)*p + (v_off_k+1)][col] = -1
        W[(w_off_k+0)*n + (w_off_i+0)][col] = 1
        W[(w_off_k+0)*n + (w_off_i+1)][col] = 1
        col += 1
        
        # M5: (A00+A01)B11 -> C00, C01
        U[(u_off_i+0)*m + (u_off_j+0)][col] = 1
        U[(u_off_i+0)*m + (u_off_j+1)][col] = 1
        V[(v_off_j+1)*p + (v_off_k+1)][col] = 1
        W[(w_off_k+0)*n + (w_off_i+0)][col] = -1
        W[(w_off_k+1)*n + (w_off_i+0)][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[(u_off_i+1)*m + (u_off_j+0)][col] = 1
        U[(u_off_i+0)*m + (u_off_j+0)][col] = -1
        V[(v_off_j+0)*p + (v_off_k+0)][col] = 1
        V[(v_off_j+0)*p + (v_off_k+1)][col] = 1
        W[(w_off_k+1)*n + (w_off_i+1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C01, C11
        U[(u_off_i+0)*m + (u_off_j+1)][col] = 1
        U[(u_off_i+1)*m + (u_off_j+1)][col] = -1
        V[(v_off_j+1)*p + (v_off_k+0)][col] = 1
        V[(v_off_j+1)*p + (v_off_k+1)][col] = 1
        W[(w_off_k+1)*n + (w_off_i+0)][col] = 1
        W[(w_off_k+1)*n + (w_off_i+1)][col] = 1
        col += 1
        
        return col

    def add_schoolbook_term(i, j, k, col):
        U[i*m + j][col] = 1
        V[j*p + k][col] = 1
        W[k*n + i][col] = 1
        return col + 1

    col = 0
    # Block 1: j in {0,1}, k in {0,1}
    col = add_strassen_block(0, 0, 0, 0, 0, 0, col)
    
    # Remainder for k in {0,1}: j=2
    col = add_schoolbook_term(0, 2, 0, col)
    col = add_schoolbook_term(1, 2, 0, col)
    col = add_schoolbook_term(0, 2, 1, col)
    col = add_schoolbook_term(1, 2, 1, col)
    
    # k=2: all j
    for i in range(n):
        for j in range(m):
            col = add_schoolbook_term(i, j, 2, col)
            
    # Block 2: j in {1,2}, k in {3,4}
    col = add_strassen_block(0, 1, 1, 3, 3, 0, col)
    
    # Remainder for k in {3,4}: j=0
    col = add_schoolbook_term(0, 0, 3, col)
    col = add_schoolbook_term(1, 0, 3, col)
    col = add_schoolbook_term(0, 0, 4, col)
    col = add_schoolbook_term(1, 0, 4, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

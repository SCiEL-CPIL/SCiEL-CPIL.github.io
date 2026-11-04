"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook decomposition for n x m by m x p
    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen decomposition for 2 x 2 by 2 x 2
    def get_strassen_2x2():
        # Strassen's 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U, V, W for 2x2 mult
        # Rows of U: (1,1), (1,2), (2,1), (2,2) -> 0, 1, 2, 3
        # Rows of V: (1,1), (1,2), (2,1), (2,2) -> 0, 1, 2, 3
        # Rows of W: (1,1), (1,2), (2,1), (2,2) -> 0, 1, 2, 3 (Trace convention W[k*n+i])
        # For 2x2: W[k*2+i]
        
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[1][0] = 1; V[2][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = -1; W[2][0] = -1; W[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1; V[2][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[0][2] = 1; W[2][2] = -1
        
        # M4
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[1][3] = 1; W[3][3] = -1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1; V[1][4] = 1
        W[1][4] = 1; W[0][4] = -1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[2][5] = 1; W[0][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[3][6] = 1; W[1][6] = 1
        
        return U, V, W

    # Global matrices
    total_rank = 0
    # Pre-calculate ranks
    r_33 = 27 # Schoolbook 3x3
    r_22 = 7  # Strassen 2x2
    r_cross = 18 + 18 + 12 + 18 + 12 + 12 # 18(12-21), 18(11-12), 12(12-22), 18(21-11), 12(22-21), 12(21-12)
    total_rank = r_33 + r_22 + r_cross
    
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    col_offset = 0
    
    # Function to add a local decomposition to global
    def add_decomposition(U_loc, V_loc, W_loc, 
                          a_i_offset, a_j_offset, 
                          b_i_offset, b_j_offset, 
                          c_i_offset, c_j_offset,
                          n_loc, m_loc, p_loc):
        nonlocal col_offset
        r = len(U_loc[0])
        
        for r_idx in range(r):
            # Map U
            for i_loc in range(n_loc):
                for j_loc in range(m_loc):
                    val = U_loc[i_loc * m_loc + j_loc][r_idx]
                    if val != 0:
                        i_glob = a_i_offset + i_loc
                        j_glob = a_j_offset + j_loc
                        U[i_glob * 5 + j_glob][col_offset + r_idx] += val
                        
            # Map V
            for j_loc in range(m_loc):
                for k_loc in range(p_loc):
                    val = V_loc[j_loc * p_loc + k_loc][r_idx]
                    if val != 0:
                        i_glob = b_i_offset + j_loc
                        j_glob = b_j_offset + k_loc
                        V[i_glob * 5 + j_glob][col_offset + r_idx] += val
                        
            # Map W
            # W trace convention: W[k*n+i, r]
            # Local W is p_loc x n_loc. Row index in local W is k_loc * n_loc + i_loc?
            # The local W from get_schoolbook/get_strassen follows W[k*n+i]
            for k_loc in range(p_loc):
                for i_loc in range(n_loc):
                    val = W_loc[k_loc * n_loc + i_loc][r_idx]
                    if val != 0:
                        i_glob = c_i_offset + i_loc
                        j_glob = c_j_offset + k_loc
                        # Wait, global W is p*n = 5*5.
                        # Global index for W is k_glob * 5 + i_glob
                        # c_i_offset is for i (row of C), c_j_offset is for k (col of C)?
                        # In C = AB, C[i, k].
                        # The task says W[k*n+i, r].
                        # So W row index is k * n + i.
                        # Here n=5.
                        # So we need k_glob and i_glob.
                        # i_glob corresponds to 'i' in C[i,k].
                        # k_glob corresponds to 'k' in C[i,k].
                        # In my mapping above:
                        # c_i_offset + i_loc -> this is i_glob
                        # c_j_offset + k_loc -> this is k_glob?
                        # Let's check the block definitions.
                        # C_11: i in 1..3, k in 1..3.
                        # c_i_offset=0, c_j_offset=0.
                        # i_glob = i_loc. k_glob = k_loc.
                        # W index = k_glob * 5 + i_glob.
                        
                        W[j_glob * 5 + i_glob][col_offset + r_idx] += val
                        
        col_offset += r

    # 1. A_11 * B_11 -> C_11 (3x3 * 3x3)
    U_33, V_33, W_33 = get_schoolbook(3, 3, 3)
    add_decomposition(U_33, V_33, W_33, 
                      0, 0, # A offset
                      0, 0, # B offset
                      0, 0, # C offset (i, k)
                      3, 3, 3)
                      
    # 2. A_22 * B_22 -> C_22 (2x2 * 2x2)
    U_22, V_22, W_22 = get_strassen_2x2()
    add_decomposition(U_22, V_22, W_22, 
                      3, 3, # A offset
                      3, 3, # B offset
                      3, 3, # C offset
                      2, 2, 2)
                      
    # 3. A_12 * B_21 -> C_11 (3x2 * 2x3)
    U_c1, V_c1, W_c1 = get_schoolbook(3, 2, 3)
    add_decomposition(U_c1, V_c1, W_c1, 
                      0, 3, # A offset (i in 0..2, j in 3..4)
                      3, 0, # B offset (i in 3..4, j in 0..2)
                      0, 0, # C offset (i in 0..2, k in 0..2)
                      3, 2, 3)
                      
    # 4. A_11 * B_12 -> C_12 (3x3 * 3x2)
    U_c2, V_c2, W_c2 = get_schoolbook(3, 3, 2)
    add_decomposition(U_c2, V_c2, W_c2, 
                      0, 0, # A offset
                      0, 3, # B offset
                      0, 3, # C offset (i in 0..2, k in 3..4)
                      3, 3, 2)
                      
    # 5. A_12 * B_22 -> C_12 (3x2 * 2x2)
    U_c3, V_c3, W_c3 = get_schoolbook(3, 2, 2)
    add_decomposition(U_c3, V_c3, W_c3, 
                      0, 3, # A offset
                      3, 3, # B offset
                      0, 3, # C offset
                      3, 2, 2)
                      
    # 6. A_21 * B_11 -> C_21 (2x3 * 3x3)
    U_c4, V_c4, W_c4 = get_schoolbook(2, 3, 3)
    add_decomposition(U_c4, V_c4, W_c4, 
                      3, 0, # A offset
                      0, 0, # B offset
                      3, 0, # C offset (i in 3..4, k in 0..2)
                      2, 3, 3)
                      
    # 7. A_22 * B_21 -> C_21 (2x2 * 2x3)
    U_c5, V_c5, W_c5 = get_schoolbook(2, 2, 3)
    add_decomposition(U_c5, V_c5, W_c5, 
                      3, 3, # A offset
                      3, 0, # B offset
                      3, 0, # C offset
                      2, 2, 3)
                      
    # 8. A_21 * B_12 -> C_22 (2x3 * 3x2)
    U_c6, V_c6, W_c6 = get_schoolbook(2, 3, 2)
    add_decomposition(U_c6, V_c6, W_c6, 
                      3, 0, # A offset
                      0, 3, # B offset
                      3, 3, # C offset
                      2, 3, 2)
                      
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

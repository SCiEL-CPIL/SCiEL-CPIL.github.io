"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    # A_local: 2x2, B_local: 2x2, C_local: 2x2
    # Offsets: A_i_off, A_j_off, B_j_off, B_k_off, C_i_off, C_k_off
    def add_strassen(A_i_off, A_j_off, B_j_off, B_k_off, C_i_off, C_k_off, col_offset):
        # U vectors for A (local indices 0,1,2,3 -> i,j)
        # V vectors for B (local indices 0,1,2,3 -> j,k)
        # W vectors for C (local indices 0,1,2,3 -> i,k)
        
        # Term 1
        U_vec = [1, 0, 0, 1]
        V_vec = [1, 1, 0, 0]
        W_vec = [1, 0, 0, 1]
        
        # Term 2
        U_vec2 = [0, 0, 1, 1]
        V_vec2 = [1, 0, 0, 0]
        W_vec2 = [0, 0, 1, -1]
        
        # Term 3
        U_vec3 = [1, 0, 0, 0]
        V_vec3 = [0, 1, 0, -1]
        W_vec3 = [0, 1, 0, 1]
        
        # Term 4
        U_vec4 = [0, 0, 0, 1]
        V_vec4 = [0, 0, 1, -1]
        W_vec4 = [1, 0, 1, 0]
        
        # Term 5
        U_vec5 = [1, 1, 0, 0]
        V_vec5 = [0, 0, 0, 1]
        W_vec5 = [-1, 1, 0, 0]
        
        # Term 6
        U_vec6 = [-1, 0, 1, 0]
        V_vec6 = [1, 1, 0, 0]
        W_vec6 = [0, 0, 0, 1]
        
        # Term 7
        U_vec7 = [0, 1, 0, -1]
        V_vec7 = [0, 0, 1, 1]
        W_vec7 = [1, 0, 0, 0]
        
        terms = [
            (U_vec, V_vec, W_vec),
            (U_vec2, V_vec2, W_vec2),
            (U_vec3, V_vec3, W_vec3),
            (U_vec4, V_vec4, W_vec4),
            (U_vec5, V_vec5, W_vec5),
            (U_vec6, V_vec6, W_vec6),
            (U_vec7, V_vec7, W_vec7)
        ]
        
        for t_idx in range(7):
            u_vals, v_vals, w_vals = terms[t_idx]
            col = col_offset + t_idx
            
            # Map U
            for li in range(2):
                for lj in range(2):
                    val = u_vals[li * 2 + lj]
                    if val != 0:
                        gi = A_i_off + li
                        gj = A_j_off + lj
                        U[gi * m + gj][col] = val
                        
            # Map V
            for lj in range(2):
                for lk in range(2):
                    val = v_vals[lj * 2 + lk]
                    if val != 0:
                        gj = B_j_off + lj
                        gk = B_k_off + lk
                        V[gj * p + gk][col] = val
                        
            # Map W
            for li in range(2):
                for lk in range(2):
                    val = w_vals[li * 2 + lk]
                    if val != 0:
                        gi = C_i_off + li
                        gk = C_k_off + lk
                        # W index is k*n + i
                        W[gk * n + gi][col] = val

    # Helper to add Schoolbook 1x4x2 block
    def add_schoolbook_1x4x2(A_row_off, B_j_off, B_k_off, C_row_off, C_k_off, col_offset):
        # A is 1x4 (row A_row_off, cols 0..3)
        # B is 4x2 (rows 0..3, cols B_k_off..B_k_off+1)
        # C is 1x2 (row C_row_off, cols C_k_off..C_k_off+1)
        
        for k_local in range(2):
            for j in range(4):
                col = col_offset + k_local * 4 + j
                
                # U: A[A_row_off, j]
                U[A_row_off * m + j][col] = 1
                
                # V: B[j, B_k_off + k_local]
                V[j * p + (B_k_off + k_local)][col] = 1
                
                # W: C[C_row_off, C_k_off + k_local]
                # Index: k*n + i
                gk = C_k_off + k_local
                gi = C_row_off
                W[gk * n + gi][col] = 1

    # Block 1a: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1b: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 2, 2, 0, 0, 0, 7)
    
    # Block 2a: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 0, 0, 2, 0, 2, 14)
    
    # Block 2b: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 2, 2, 2, 0, 2, 21)
    
    # Block 3: A[2, 0:4] * B[0:4, 0:2] -> C[2, 0:2]
    add_schoolbook_1x4x2(2, 0, 0, 2, 0, 28)
    
    # Block 4: A[2, 0:4] * B[0:4, 2:4] -> C[2, 2:4]
    add_schoolbook_1x4x2(2, 0, 2, 2, 2, 36)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

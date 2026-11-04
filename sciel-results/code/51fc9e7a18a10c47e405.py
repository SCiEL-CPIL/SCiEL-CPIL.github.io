"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a rank 112 decomposition based on 8 Strassen multiplications for 4x4x4
    # and two such blocks for 4x4x8.
    
    # 1. Generate Strassen factors for 2x2x2 multiplication
    # U, V, W are 4x7 matrices
    U_str = [[0]*7 for _ in range(4)]
    V_str = [[0]*7 for _ in range(4)]
    W_str = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    U_str[0][0] = 1; U_str[3][0] = 1
    V_str[0][0] = 1; V_str[3][0] = 1
    W_str[0][0] = 1; W_str[3][0] = 1
    
    # M2 = (c+d)e
    U_str[2][1] = 1; U_str[3][1] = 1
    V_str[0][1] = 1
    W_str[2][1] = 1; W_str[3][1] = -1
    
    # M3 = a(f-h)
    U_str[0][2] = 1
    V_str[1][2] = 1; V_str[3][2] = -1
    W_str[1][2] = 1; W_str[3][2] = 1
    
    # M4 = d(g-f)
    U_str[3][3] = 1
    V_str[2][3] = 1; V_str[1][3] = -1
    W_str[0][3] = 1; W_str[2][3] = 1
    
    # M5 = (a+b)h
    U_str[0][4] = 1; U_str[1][4] = 1
    V_str[3][4] = 1
    W_str[0][4] = -1; W_str[1][4] = 1
    
    # M6 = (c-a)(e+f)
    U_str[2][5] = 1; U_str[0][5] = -1
    V_str[0][5] = 1; V_str[1][5] = 1
    W_str[3][5] = 1
    
    # M7 = (b-d)(g+h)
    U_str[1][6] = 1; U_str[3][6] = -1
    V_str[2][6] = 1; V_str[3][6] = 1
    W_str[0][6] = 1; W_str[2][6] = 1
    
    # 2. Construct 4x4x4 factors (Rank 56)
    # 8 Strassen blocks for each 4x4x4 multiplication
    # U_44: 16x56, V_44: 16x56, W_44: 16x56
    U_44 = [[0]*56 for _ in range(16)]
    V_44 = [[0]*56 for _ in range(16)]
    W_44 = [[0]*56 for _ in range(16)]
    
    # Block mapping for 4x4 matrix composed of 2x2 blocks
    # A[I][K] where I,K in {0,1}
    # Global row for A: i*4+k
    # Global row for B: k*4+l
    # Global row for C: i*4+l
    
    block_idx = 0
    for I in range(2):
        for K in range(2):
            for L in range(2):
                # We are computing term A[I][K] * B[K][L] which contributes to C[I][L]
                # This is one of the 8 Strassen multiplications
                # Note: The sum over K is handled by the linearity of the tensor decomposition.
                # Each (I, K, L) triple corresponds to one 2x2x2 multiplication.
                # There are 2*2*2 = 8 such multiplications.
                
                # Local to Global mapping
                # A indices: i_loc, k_loc
                # i_glob = 2*I + i_loc
                # k_glob = 2*K + k_loc
                # A_global_row = i_glob * 4 + k_glob
                
                # B indices: k_loc, l_loc
                # k_glob = 2*K + k_loc
                # l_glob = 2*L + l_loc
                # B_global_row = k_glob * 4 + l_glob
                
                # C indices: i_loc, l_loc
                # i_glob = 2*I + i_loc
                # l_glob = 2*L + l_loc
                # C_global_row = i_glob * 4 + l_glob
                
                for r in range(7):
                    col = block_idx * 7 + r
                    for u_loc in range(4):
                        if U_str[u_loc][r] != 0:
                            i_loc, k_loc = divmod(u_loc, 2)
                            i_glob = 2*I + i_loc
                            k_glob = 2*K + k_loc
                            u_glob = i_glob * 4 + k_glob
                            U_44[u_glob][col] += U_str[u_loc][r]
                            
                    for v_loc in range(4):
                        if V_str[v_loc][r] != 0:
                            k_loc, l_loc = divmod(v_loc, 2)
                            k_glob = 2*K + k_loc
                            l_glob = 2*L + l_loc
                            v_glob = k_glob * 4 + l_glob
                            V_44[v_glob][col] += V_str[v_loc][r]
                            
                    for w_loc in range(4):
                        if W_str[w_loc][r] != 0:
                            i_loc, l_loc = divmod(w_loc, 2)
                            i_glob = 2*I + i_loc
                            l_glob = 2*L + l_loc
                            w_glob = i_glob * 4 + l_glob
                            W_44[w_glob][col] += W_str[w_loc][r]
                            
                block_idx += 1
    # block_idx should be 8
    
    # 3. Construct 4x4x8 factors (Rank 112)
    # U_448: 16x112, V_448: 32x112, W_448: 32x112
    U_448 = [[0]*112 for _ in range(16)]
    V_448 = [[0]*112 for _ in range(32)]
    W_448 = [[0]*112 for _ in range(32)]
    
    # First 56 columns: A * B1 (B1 is first 4 cols of B)
    for i in range(16):
        for c in range(56):
            U_448[i][c] = U_44[i][c]
            V_448[i][c] = V_44[i][c] # B1 rows are 0-15
            W_448[i][c] = W_44[i][c] # C1 rows are 0-15
    
    # Next 56 columns: A * B2 (B2 is last 4 cols of B)
    for i in range(16):
        for c in range(56):
            U_448[i][c + 56] = U_44[i][c]
            # B2 rows are 16-31. Local row i in V_44 corresponds to global row 16+i
            V_448[i + 16][c + 56] = V_44[i][c]
            # C2 rows are 16-31. Local row i in W_44 corresponds to global row 16+i
            W_448[i + 16][c + 56] = W_44[i][c]
            
    return {"U": U_448, "V": V_448, "W": W_448}
# EVOLVE-BLOCK-END

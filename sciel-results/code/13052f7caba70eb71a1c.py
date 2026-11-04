"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a rank-28 decomposition based on 4 independent 2x2x2 Strassen decompositions.
    # A is 2x4, split into A_L (2x2) and A_R (2x2).
    # B is 4x4, split into 4 2x2 blocks: B_TL, B_TR, B_BL, B_BR.
    # C[:, 0:2] = A_L @ B_TL + A_R @ B_BL
    # C[:, 2:4] = A_L @ B_TR + A_R @ B_BR
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Helper to add a Strassen decomposition for X (2x2) * Y (2x2) = Z (2x2)
    # X indices in U: x_base + [0, 1, 2, 3] (row-major 2x2)
    # Y indices in V: y_base + [0, 1, 2, 3]
    # Z indices in W: z_base + [0, 1, 2, 3]
    # Offset in rank: r_off
    
    def add_strassen(x_base, y_base, z_base, r_off):
        # X is 2x2, indices: 00, 01, 10, 11 -> x_base+0, x_base+1, x_base+2, x_base+3
        # Y is 2x2, indices: 00, 01, 10, 11 -> y_base+0, y_base+1, y_base+2, y_base+3
        # Z is 2x2, indices: 00, 01, 10, 11 -> z_base+0, z_base+1, z_base+2, z_base+3
        
        # M1 = (X00 + X11) * (Y00 + Y11)
        U[x_base + 0][r_off + 0] += 1
        U[x_base + 3][r_off + 0] += 1
        V[y_base + 0][r_off + 0] += 1
        V[y_base + 3][r_off + 0] += 1
        W[z_base + 0][r_off + 0] += 1
        W[z_base + 3][r_off + 0] += 1
        
        # M2 = (X10 + X11) * Y00
        U[x_base + 2][r_off + 1] += 1
        U[x_base + 3][r_off + 1] += 1
        V[y_base + 0][r_off + 1] += 1
        W[z_base + 2][r_off + 1] += 1
        W[z_base + 3][r_off + 1] += 1
        
        # M3 = X00 * (Y01 - Y11)
        U[x_base + 0][r_off + 2] += 1
        V[y_base + 1][r_off + 2] += 1
        V[y_base + 3][r_off + 2] -= 1
        W[z_base + 1][r_off + 2] += 1
        W[z_base + 3][r_off + 2] -= 1
        
        # M4 = X11 * (Y10 - Y00)
        U[x_base + 3][r_off + 3] += 1
        V[y_base + 2][r_off + 3] += 1
        V[y_base + 0][r_off + 3] -= 1
        W[z_base + 0][r_off + 3] -= 1
        W[z_base + 2][r_off + 3] += 1
        
        # M5 = (X00 + X01) * Y11
        U[x_base + 0][r_off + 4] += 1
        U[x_base + 1][r_off + 4] += 1
        V[y_base + 3][r_off + 4] += 1
        W[z_base + 1][r_off + 4] += 1
        W[z_base + 3][r_off + 4] += 1
        
        # M6 = (X10 - X00) * (Y00 + Y01)
        U[x_base + 2][r_off + 5] += 1
        U[x_base + 0][r_off + 5] -= 1
        V[y_base + 0][r_off + 5] += 1
        V[y_base + 1][r_off + 5] += 1
        W[z_base + 0][r_off + 5] += 1
        W[z_base + 2][r_off + 5] += 1
        
        # M7 = (X01 - X11) * (Y10 + Y11)
        U[x_base + 1][r_off + 6] += 1
        U[x_base + 3][r_off + 6] -= 1
        V[y_base + 2][r_off + 6] += 1
        V[y_base + 3][r_off + 6] += 1
        W[z_base + 0][r_off + 6] += 1
        W[z_base + 1][r_off + 6] += 1

    # Block 1: A_L (cols 0-1 of A) * B_TL (rows 0-1, cols 0-1 of B) -> C[:, 0:2] part 1
    # A indices: 0,1 (row 0), 2,3 (row 1) for A_L?
    # A is 2x4. Row 0: A[0,0], A[0,1], A[0,2], A[0,3]. Row 1: A[1,0], A[1,1], A[1,2], A[1,3].
    # U index for A[i, j] is i * m + j = i * 4 + j.
    # A_L uses j=0,1.
    # A_L[0,0] -> U[0], A_L[0,1] -> U[1], A_L[1,0] -> U[4], A_L[1,1] -> U[5].
    # So x_base for A_L is not contiguous in U.
    # My add_strassen assumes contiguous 2x2 block in memory.
    # I need to map the specific indices.
    
    # Let's rewrite add_strassen to take explicit index lists.
    # U_indices: [u00, u01, u10, u11]
    # V_indices: [v00, v01, v10, v11]
    # W_indices: [w00, w01, w10, w11]
    
    def add_strassen_explicit(u_idx, v_idx, w_idx, r_off):
        # u_idx: [00, 01, 10, 11]
        # v_idx: [00, 01, 10, 11]
        # w_idx: [00, 01, 10, 11]
        
        # M1
        U[u_idx[0]][r_off + 0] += 1
        U[u_idx[3]][r_off + 0] += 1
        V[v_idx[0]][r_off + 0] += 1
        V[v_idx[3]][r_off + 0] += 1
        W[w_idx[0]][r_off + 0] += 1
        W[w_idx[3]][r_off + 0] += 1
        
        # M2
        U[u_idx[2]][r_off + 1] += 1
        U[u_idx[3]][r_off + 1] += 1
        V[v_idx[0]][r_off + 1] += 1
        W[w_idx[2]][r_off + 1] += 1
        W[w_idx[3]][r_off + 1] += 1
        
        # M3
        U[u_idx[0]][r_off + 2] += 1
        V[v_idx[1]][r_off + 2] += 1
        V[v_idx[3]][r_off + 2] -= 1
        W[w_idx[1]][r_off + 2] += 1
        W[w_idx[3]][r_off + 2] -= 1
        
        # M4
        U[u_idx[3]][r_off + 3] += 1
        V[v_idx[2]][r_off + 3] += 1
        V[v_idx[0]][r_off + 3] -= 1
        W[w_idx[0]][r_off + 3] -= 1
        W[w_idx[2]][r_off + 3] += 1
        
        # M5
        U[u_idx[0]][r_off + 4] += 1
        U[u_idx[1]][r_off + 4] += 1
        V[v_idx[3]][r_off + 4] += 1
        W[w_idx[1]][r_off + 4] += 1
        W[w_idx[3]][r_off + 4] += 1
        
        # M6
        U[u_idx[2]][r_off + 5] += 1
        U[u_idx[0]][r_off + 5] -= 1
        V[v_idx[0]][r_off + 5] += 1
        V[v_idx[1]][r_off + 5] += 1
        W[w_idx[0]][r_off + 5] += 1
        W[w_idx[2]][r_off + 5] += 1
        
        # M7
        U[u_idx[1]][r_off + 6] += 1
        U[u_idx[3]][r_off + 6] -= 1
        V[v_idx[2]][r_off + 6] += 1
        V[v_idx[3]][r_off + 6] += 1
        W[w_idx[0]][r_off + 6] += 1
        W[w_idx[1]][r_off + 6] += 1

    # Define indices
    # A: 2x4. U index = i*4 + j.
    # A_L: i in 0,1; j in 0,1.
    # u_A_L = [0, 1, 4, 5] (00, 01, 10, 11)
    # A_R: i in 0,1; j in 2,3.
    # u_A_R = [2, 3, 6, 7]
    
    # B: 4x4. V index = i*4 + j.
    # B_TL: i in 0,1; j in 0,1.
    # v_B_TL = [0, 1, 4, 5]
    # B_TR: i in 0,1; j in 2,3.
    # v_B_TR = [2, 3, 6, 7]
    # B_BL: i in 2,3; j in 0,1.
    # v_B_BL = [8, 9, 12, 13]
    # B_BR: i in 2,3; j in 2,3.
    # v_B_BR = [10, 11, 14, 15]
    
    # C: 2x4. W index = k*n + i = k*2 + i.
    # C[:, 0:2] corresponds to k=0,1.
    # w_C_L = [0, 1, 2, 3] (k=0: i=0,1 -> 0,1; k=1: i=0,1 -> 2,3)
    # Wait, W[k*n + i].
    # k=0: W[0] (i=0), W[1] (i=1).
    # k=1: W[2] (i=0), W[3] (i=1).
    # So w_C_L = [0, 1, 2, 3].
    
    # C[:, 2:4] corresponds to k=2,3.
    # k=2: W[4] (i=0), W[5] (i=1).
    # k=3: W[6] (i=0), W[7] (i=1).
    # So w_C_R = [4, 5, 6, 7].
    
    # Block 1: A_L * B_TL -> adds to C_L
    add_strassen_explicit(u_A_L, v_B_TL, w_C_L, 0)
    
    # Block 2: A_R * B_BL -> adds to C_L
    add_strassen_explicit(u_A_R, v_B_BL, w_C_L, 7)
    
    # Block 3: A_L * B_TR -> adds to C_R
    add_strassen_explicit(u_A_L, v_B_TR, w_C_R, 14)
    
    # Block 4: A_R * B_BR -> adds to C_R
    add_strassen_explicit(u_A_R, v_B_BR, w_C_R, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

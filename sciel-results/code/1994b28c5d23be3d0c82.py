def construct():
    n, m, p = 3, 4, 8
    # Build rank-64 decomposition
    # 2×4×8 part: 56 terms
    # 1×4×8 part: 8 terms
    # Total: 64
    
    R = 64
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # Strassen factors for 2×2×2
    # Each factor: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict from (i,j) to value, for A[i,j]
    # v_coeffs: dict from (j,k) to value, for B[j,k]  
    # w_coeffs: dict from (k,i) to value, for C[k,i] -> C[i,k]
    
    strassen = [
        # (A coeffs, B coeffs, C coeffs)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    def add_strassen_block(i_off, j_off, k_off):
        """Add Strassen decomposition for a 2×2×2 block at offset (i_off, j_off, k_off).
        i_off: offset for row index of A (and C)
        j_off: offset for col index of A / row index of B
        k_off: offset for col index of B (and C)
        Returns number of columns used.
        """
        global col
        for (a_coeffs, b_coeffs, c_coeffs) in strassen:
            for (ai, aj), au in a_coeffs.items():
                U[(i_off+ai)*m + (j_off+aj), col] += au
            for (bi, bk), bv in b_coeffs.items():
                V[(j_off+bi)*p + (k_off+bk), col] += bv
            for (ck, ci), cw in c_coeffs.items():
                W[(k_off+ck)*n + (i_off+ci), col] += cw
            col += 1
    
    # 2×4×8 part (i=0,1):
    # Split into two 2×2×8 blocks (j=0,1 and j=2,3)
    # Each 2×2×8 splits into two 2×2×4 blocks
    # Each 2×2×4 splits into two 2×2×2 blocks
    
    # Block 1: i=0,1; j=0,1; k=0,1
    add_strassen_block(0, 0, 0)  # 7 terms
    # Block 2: i=0,1; j=0,1; k=2,3
    add_strassen_block(0, 0, 2)  # 7 terms
    # Block 3: i=0,1; j=0,1; k=4,5
    add_strassen_block(0, 0, 4)  # 7 terms
    # Block 4: i=0,1; j=0,1; k=6,7
    add_strassen_block(0, 0, 6)  # 7 terms
    # Block 5: i=0,1; j=2,3; k=0,1
    add_strassen_block(0, 2, 0)  # 7 terms
    # Block 6: i=0,1; j=2,3; k=2,3
    add_strassen_block(0, 2, 2)  # 7 terms
    # Block 7: i=0,1; j=2,3; k=4,5
    add_strassen_block(0, 2, 4)  # 7 terms
    # Block 8: i=0,1; j=2,3; k=6,7
    add_strassen_block(0, 2, 6)  # 7 terms
    # Total: 56 terms
    
    # 1×4×8 part (i=2):
    # T[j,k] = 1 for j=0..3, k=0..7
    # Use rank 8: for each k, one term
    for k in range(8):
        for j in range(4):
            U[2*m + j, col] = 1.0
        V[j*p + k, col] = 1.0  # Wait, this is wrong
        W[k*n + 2, col] = 1.0
        col += 1
    
    # Hmm, the 1×4×8 part needs more care
    # T[j,k] = sum_r u_{r,j} v_{r,j,k} w_{r,k} = 1
    # With rank 8, one term per k:
    # For term r=k: u_{k,j} = 1 for all j, v_{k,j,k} = 1 for all j, w_{k,k} = 1
    # But v is indexed by (j*p + k), so v_{j*p+k, r}
    
    # Let me redo this
    # Actually, I made an error. Let me restart the 1×4×8 part.

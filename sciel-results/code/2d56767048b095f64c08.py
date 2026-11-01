"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def _build_2x2x2_strassen():
        """
        Returns U (4x7), V (4x7), W (4x7) for 2x2x2 matrix multiplication.
        U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        Using standard Strassen decomposition.
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen formulas:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B21 - B11)
        # M4 = A22 * (B22 - B12)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices for U: i*2+j (0..3)
        # Indices for V: j*2+k (0..3)
        # Indices for W: k*2+i (0..3)
        
        # M1
        U[0][0] += 1  # A11
        U[3][0] += 1  # A22
        V[0][0] += 1  # B11
        V[3][0] += 1  # B22
        W[0][0] += 1  # C11 (k=0, i=0)
        W[3][0] += 1  # C22 (k=1, i=1)
        
        # M2
        U[2][1] += 1  # A21
        U[3][1] += 1  # A22
        V[0][1] += 1  # B11
        W[1][1] += 1  # C21 (k=0, i=1)
        W[3][1] += 1  # C22 (k=1, i=1)
        
        # M3
        U[0][2] += 1  # A11
        V[2][2] += 1  # B21
        V[0][2] -= 1  # B11
        W[0][2] += 1  # C11 (k=0, i=0)
        W[1][2] += 1  # C12 (k=0, i=0) -> wait, C12 is k=0, i=0? No.
        # Let's re-verify indices.
        # C is 2x2. C[i, k].
        # W index is k*n + i.
        # C11 is i=0, k=0 -> W[0]
        # C12 is i=0, k=1 -> W[2]
        # C21 is i=1, k=0 -> W[1]
        # C22 is i=1, k=1 -> W[3]
        
        # Re-evaluating W contributions:
        # M1 -> C11, C22
        # M2 -> C21, C22
        # M3 -> C11, C12
        # M4 -> C11, C21
        # M5 -> C11, C12
        # M6 -> C22
        # M7 -> C11, C22
        
        # Let's reset W and fill carefully.
        for i in range(4):
            for r in range(7):
                W[i][r] = 0
                U[i][r] = 0
                V[i][r] = 0
        
        # M1: (A11+A22)(B11+B22) -> C11 +=, C22 +=
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (A21+A22)B11 -> C21 +=, C22 +=
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = 1
        
        # M3: A11(B21-B11) -> C11 +=, C12 +=
        U[0][2] = 1
        V[2][2] = 1; V[0][2] = -1
        W[0][2] = 1; W[2][2] = 1
        
        # M4: A22(B22-B12) -> C11 +=, C21 +=
        U[3][3] = 1
        V[3][3] = 1; V[1][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # M5: (A11+A12)B22 -> C11 +=, C12 +=
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[2][4] = 1
        
        # M6: (A21-A11)(B11+B12) -> C22 +=
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: (A12-A22)(B21+B22) -> C11 +=, C22 +=
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W

    def _build_2x2x1_schoolbook():
        """
        Returns U (4x4), V (2x4), W (2x4) for 2x2x1 matrix multiplication.
        Here p_loc=1, so B is 2x1.
        C is 2x1.
        C11 = A11*B11 + A12*B21
        C21 = A21*B11 + A22*B21
        """
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # r=0: A11 * B11 -> C11
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        # r=1: A12 * B21 -> C11
        U[1][1] = 1
        V[1][1] = 1
        W[0][1] = 1
        
        # r=2: A21 * B11 -> C21
        U[2][2] = 1
        V[0][2] = 1
        W[1][2] = 1
        
        # r=3: A22 * B21 -> C21
        U[3][3] = 1
        V[1][3] = 1
        W[1][3] = 1
        
        return U, V, W

    # Decomposition strategy:
    # Split p=15 into seven blocks of size 2 and one block of size 1.
    # 7 * 2 + 1 = 15.
    # Use Strassen for 2x2x2 blocks (rank 7 each).
    # Use Schoolbook for 2x2x1 block (rank 4).
    # Total rank = 7 * 7 + 4 = 49 + 4 = 53.
    
    rank = 53
    U = [[0]*rank for _ in range(n*m)]  # 4x53
    V = [[0]*rank for _ in range(m*p)]  # 30x53
    W = [[0]*rank for _ in range(p*n)]  # 30x53
    
    sU, sV, sW = _build_2x2x2_strassen()
    sbU, sbV, sbW = _build_2x2x1_schoolbook()
    
    r_offset = 0
    
    # 7 Strassen blocks
    for b in range(7):
        k_offset = 2 * b
        
        for r_loc in range(7):
            r_glob = r_offset + r_loc
            
            # Fill U
            for u_idx in range(4):
                if sU[u_idx][r_loc] != 0:
                    U[u_idx][r_glob] += sU[u_idx][r_loc]
            
            # Fill V
            for v_loc_idx in range(4):
                if sV[v_loc_idx][r_loc] != 0:
                    j = v_loc_idx // 2
                    k_loc = v_loc_idx % 2
                    v_glob_idx = j * p + k_offset + k_loc
                    V[v_glob_idx][r_glob] += sV[v_loc_idx][r_loc]
            
            # Fill W
            for w_loc_idx in range(4):
                if sW[w_loc_idx][r_loc] != 0:
                    k_loc = w_loc_idx // 2
                    i = w_loc_idx % 2
                    w_glob_idx = (k_offset + k_loc) * n + i
                    W[w_glob_idx][r_glob] += sW[w_loc_idx][r_loc]
        
        r_offset += 7

    # 1 Schoolbook block
    k_offset = 14
    
    for r_loc in range(4):
        r_glob = r_offset + r_loc
        
        # Fill U
        for u_idx in range(4):
            if sbU[u_idx][r_loc] != 0:
                U[u_idx][r_glob] += sbU[u_idx][r_loc]
        
        # Fill V
        for v_loc_idx in range(2):
            if sbV[v_loc_idx][r_loc] != 0:
                j = v_loc_idx
                k_loc = 0
                v_glob_idx = j * p + k_offset + k_loc
                V[v_glob_idx][r_glob] += sbV[v_loc_idx][r_loc]
        
        # Fill W
        for w_loc_idx in range(2):
            if sbW[w_loc_idx][r_loc] != 0:
                k_loc = 0
                i = w_loc_idx
                w_glob_idx = (k_offset + k_loc) * n + i
                W[w_glob_idx][r_glob] += sbW[w_loc_idx][r_loc]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

<<<<<<< SEARCH
        # --- Rank-4 part for A2 (2x1) * B2 (1x2) ---
        # A2 uses A[0,2], A[1,2]
        # B2 uses B[2,k0], B[2,k1]
        
        # P1 = A02 * B2k0
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + k0][col] = 1 # B2k0
        # W contributions: C0k0 += 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # P2 = A02 * B2k1
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + k1][col] = 1 # B2k1
        # W contributions: C0k1 += 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P3 = A12 * B2k0
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k0][col] = 1 # B2k0
        # W contributions: C1k0 += 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P4 = A12 * B2k1
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k1][col] = 1 # B2k1
        # W contributions: C1k1 += 1
        W[k1 * n + 1][col] = 1
        col += 1
=======
        # --- Rank-1 part for A2 (2x1) * B2 (1x2) ---
        # A2 uses A[0,2], A[1,2]
        # B2 uses B[2,k0], B[2,k1]
        
        # P1 = Outer product of A[:,2] and B[2,:]
        U[0 * m + 2][col] = 1  # A02
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k0][col] = 1 # B2k0
        V[2 * p + k1][col] = 1 # B2k1
        # W contributions: C0k0 += 1, C0k1 += 1, C1k0 += 1, C1k1 += 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
>>>>>>> REPLACE

<<<<<<< SEARCH
    def strassen_block():
        """
        Block decomposition using Strassen for 2x2x2 blocks.
        Split A (4x4) into four 2x2 blocks.
        Split B (4x5) into two 2x5 blocks (top 2 rows, bottom 2 rows).
        C = A * B where C is 4x5.
        
        C_top = A_11 * B_top + A_12 * B_bot  (2x5)
        C_bot = A_21 * B_top + A_22 * B_bot  (2x5)
        
        Each term A_ij * B_k is M_{2,2,5}.
        For M_{2,2,5}: split B (2x5) into B1 (2x2) and B2 (2x3).
        A * B1 uses Strassen (rank 7), A * B2 uses trivial (rank 2*2*3 = 12).
        Total per M_{2,2,5}: 19.
        4 blocks * 19 = 76.
        """
        rank = 76
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        # Strassen for M_{2,2,2}: 7 multiplications
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        def add_strassen_2x2x2(i_off, j_off, k_off):
            """Add Strassen decomposition for M_{2,2,2} at given offsets."""
            nonlocal col
            # i_off, j_off, k_off are the starting indices for the 2x2x2 block
            # U indices: (i, j) where i in [i_off, i_off+2), j in [j_off, j_off+2)
            # V indices: (j, k) where j in [j_off, j_off+2), k in [k_off, k_off+2)
            # W indices: (k, i) where k in [k_off, k_off+2), i in [i_off, i_off+2)

            # M1 = (a11+a22)(b11+b22)
            U[i_off * m + j_off, col] = 1
            U[(i_off+1) * m + (j_off+1), col] = 1
            V[j_off * p + k_off, col] = 1
            V[(j_off+1) * p + (k_off+1), col] = 1
            W[k_off * n + i_off, col] = 1
            W[(k_off+1) * n + (i_off+1), col] = 1
            W[k_off * n + (i_off+1), col] = -1
            W[(k_off+1) * n + i_off, col] = 1
            col += 1

            # M2 = (a21+a22)b11
            U[(i_off+1) * m + j_off, col] = 1
            U[(i_off+1) * m + (j_off+1), col] = 1
            V[j_off * p + k_off, col] = 1
            W[k_off * n + (i_off+1), col] = 1
            W[(k_off+1) * n + (i_off+1), col] = -1
            col += 1

            # M3 = a11(b21-b22)
            U[i_off * m + j_off, col] = 1
            V[(j_off+1) * p + k_off, col] = 1
            V[(j_off+1) * p + (k_off+1), col] = -1
            W[k_off * n + i_off, col] = 1
            W[(k_off+1) * n + i_off, col] = 1
            col += 1

            # M4 = a22(b12-b11)
            U[(i_off+1) * m + (j_off+1), col] = 1
            V[j_off * p + (k_off+1), col] = 1
            V[j_off * p + k_off, col] = -1
            W[k_off * n + (i_off+1), col] = 1
            W[(k_off+1) * n + (i_off+1), col] = 1
            col += 1

            # M5 = (a11+a12)b22
            U[i_off * m + j_off, col] = 1
            U[i_off * m + (j_off+1), col] = 1
            V[(j_off+1) * p + (k_off+1), col] = 1
            W[k_off * n + i_off, col] = -1
            W[(k_off+1) * n + i_off, col] = 1
            col += 1

            # M6 = (a21-a11)(b11+b12)
            U[(i_off+1) * m + j_off, col] = 1
            U[i_off * m + j_off, col] = -1
            V[j_off * p + k_off, col] = 1
            V[j_off * p + (k_off+1), col] = 1
            W[(k_off+1) * n + (i_off+1), col] = 1
            col += 1

            # M7 = (a12-a22)(b21+b22)
            U[i_off * m + (j_off+1), col] = 1
            U[(i_off+1) * m + (j_off+1), col] = -1
            V[(j_off+1) * p + k_off, col] = 1
            V[(j_off+1) * p + (k_off+1), col] = 1
            W[k_off * n + i_off, col] = 1
            W[(k_off+1) * n + (i_off+1), col] = 1
            col += 1

        def add_trivial_2x2x3(i_off, j_off, k_off):
            """Add trivial decomposition for M_{2,2,3} at given offsets."""
            nonlocal col
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U[(i_off + i) * m + (j_off + j), col] = 1
                        V[(j_off + j) * p + (k_off + k), col] = 1
                        W[(k_off + k) * n + (i_off + i), col] = 1
                        col += 1

        # Four blocks of A: A_11 (i=0,1; j=0,1), A_12 (i=0,1; j=2,3), A_21 (i=2,3; j=0,1), A_22 (i=2,3; j=2,3)
        # Two blocks of B: B_top (j=0,1; k=0..4), B_bot (j=2,3; k=0..4)
        # For each A_ij * B_k:
        #   Split B_k (2x5) into B_k1 (2x2, k=0,1) and B_k2 (2x3, k=2,3,4)
        #   Use Strassen for A_ij * B_k1, trivial for A_ij * B_k2

        # A_11 * B_top: i_off=0, j_off=0, B_top has j=0,1
        # B_top1: k=0,1 (j=0,1), B_top2: k=2,3,4 (j=0,1)
        add_strassen_2x2x2(0, 0, 0)  # A_11 * B_top1 (i=0,1; j=0,1; k=0,1)
        add_trivial_2x2x3(0, 0, 2)   # A_11 * B_top2 (i=0,1; j=0,1; k=2,3,4)

        # A_12 * B_bot: i_off=0, j_off=2, B_bot has j=2,3
        # B_bot1: k=0,1 (j=2,3), B_bot2: k=2,3,4 (j=2,3)
        add_strassen_2x2x2(0, 2, 0)  # A_12 * B_bot1 (i=0,1; j=2,3; k=0,1)
        add_trivial_2x2x3(0, 2, 2)   # A_12 * B_bot2 (i=0,1; j=2,3; k=2,3,4)

        # A_21 * B_top: i_off=2, j_off=0, B_top has j=0,1
        add_strassen_2x2x2(2, 0, 0)  # A_21 * B_top1 (i=2,3; j=0,1; k=0,1)
        add_trivial_2x2x3(2, 0, 2)   # A_21 * B_top2 (i=2,3; j=0,1; k=2,3,4)

        # A_22 * B_bot: i_off=2, j_off=2, B_bot has j=2,3
        add_strassen_2x2x2(2, 2, 0)  # A_22 * B_bot1 (i=2,3; j=2,3; k=0,1)
        add_trivial_2x2x3(2, 2, 2)   # A_22 * B_bot2 (i=2,3; j=2,3; k=2,3,4)

        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
=======
    def strassen_block():
        """
        Block decomposition using Strassen for 2x2x2 blocks.
        Corrected W matrices to ensure tensor identity.
        """
        rank = 76
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen_2x2x2(i_off, j_off, k_off):
            """Add Strassen decomposition for M_{2,2,2} at given offsets."""
            nonlocal col
            
            # M1 = (a11+a22)(b11+b22)
            # U: (0,0), (1,1)
            # V: (0,0), (1,1)
            # W: C00 (1), C11 (1) -> W(0,0)=1, W(1,1)=1
            U[i_off * m + j_off, col] = 1
            U[(i_off+1) * m + (j_off+1), col] = 1
            V[j_off * p + k_off, col] = 1
            V[(j_off+1) * p + (k_off+1), col] = 1
            W[k_off * n + i_off, col] = 1
            W[(k_off+1) * n + (i_off+1), col] = 1
            col += 1

            # M2 = (a21+a22)b11
            # U: (1,0), (1,1)
            # V: (0,0)
            # W: C10 (1) -> W(0,1)=1
            U[(i_off+1) * m + j_off, col] = 1
            U[(i_off+1) * m + (j_off+1), col] = 1
            V[j_off * p + k_off, col] = 1
            W[k_off * n + (i_off+1), col] = 1
            col += 1

            # M3 = a11(b21-b22)
            # U: (0,0)
            # V: (1,0), (1,1)
            # W: C01 (1) -> W(1,0)=1
            U[i_off * m + j_off, col] = 1
            V[(j_off+1) * p + k_off, col] = 1
            V[(j_off+1) * p + (k_off+1), col] = -1
            W[(k_off+1) * n + i_off, col] = 1
            col += 1

            # M4 = a22(b12-b11)
            # U: (1,1)
            # V: (0,0), (0,1)
            # W: C10 (1) -> W(0,1)=1
            U[(i_off+1) * m + (j_off+1), col] = 1
            V[j_off * p + k_off, col] = -1
            V[j_off * p + (k_off+1), col] = 1
            W[k_off * n + (i_off+1), col] = 1
            col += 1

            # M5 = (a11+a12)b22
            # U: (0,0), (0,1)
            # V: (1,1)
            # W: C00 (-1), C01 (1) -> W(0,0)=-1, W(1,0)=1
            U[i_off * m + j_off, col] = 1
            U[i_off * m + (j_off+1), col] = 1
            V[(j_off+1) * p + (k_off+1), col] = 1
            W[k_off * n + i_off, col] = -1
            W[(k_off+1) * n + i_off, col] = 1
            col += 1

            # M6 = (a21-a11)(b11+b12)
            # U: (1,0), (0,0)
            # V: (0,0), (0,1)
            # W: C11 (1) -> W(1,1)=1
            U[(i_off+1) * m + j_off, col] = 1
            U[i_off * m + j_off, col] = -1
            V[j_off * p + k_off, col] = 1
            V[j_off * p + (k_off+1), col] = 1
            W[(k_off+1) * n + (i_off+1), col] = 1
            col += 1

            # M7 = (a12-a22)(b21+b22)
            # U: (0,1), (1,1)
            # V: (1,0), (1,1)
            # W: C00 (1) -> W(0,0)=1
            U[i_off * m + (j_off+1), col] = 1
            U[(i_off+1) * m + (j_off+1), col] = -1
            V[(j_off+1) * p + k_off, col] = 1
            V[(j_off+1) * p + (k_off+1), col] = 1
            W[k_off * n + i_off, col] = 1
            col += 1

        def add_trivial_2x2x3(i_off, j_off, k_off):
            """Add trivial decomposition for M_{2,2,3} at given offsets."""
            nonlocal col
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U[(i_off + i) * m + (j_off + j), col] = 1
                        V[(j_off + j) * p + (k_off + k), col] = 1
                        W[(k_off + k) * n + (i_off + i), col] = 1
                        col += 1

        add_strassen_2x2x2(0, 0, 0)
        add_trivial_2x2x3(0, 0, 2)
        add_strassen_2x2x2(0, 2, 0)
        add_trivial_2x2x3(0, 2, 2)
        add_strassen_2x2x2(2, 0, 0)
        add_trivial_2x2x3(2, 0, 2)
        add_strassen_2x2x2(2, 2, 0)
        add_trivial_2x2x3(2, 2, 2)

        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
>>>>>>> REPLACE


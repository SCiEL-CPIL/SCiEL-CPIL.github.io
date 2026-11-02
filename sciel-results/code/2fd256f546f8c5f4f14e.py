# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 8 matrix.
    """
    n, m, p = DIMENSIONS
    
    def schoolbook():
        """Return the schoolbook decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify_full(U, V, W, rank, n, m, p):
        """Verify full tensor decomposition."""
        for i in range(n):
            for k in range(p):
                for i_p in range(n):
                    for j_p in range(m):
                        for j_pp in range(m):
                            for k_pp in range(p):
                                total = 0
                                for c in range(rank):
                                    total += U[i_p * m + j_p][c] * V[j_pp * p + k_pp][c] * W[k_pp * n + i][c]
                                expected = 1 if (i_p == i and k_pp == k and j_p == j_pp) else 0
                                if total != expected:
                                    return False
        return True

    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen (Rank 7).
        U is 4x7, V is 4x7, W is 4x7.
        Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        # Coefficients for Strassen
        # M0 = (A00+A11)(B00+B11)
        # M1 = (A10+A11)B00
        # M2 = A00(B10-B11)
        # M3 = A11(B00-B01)
        # M4 = (A00+A01)B11
        # M5 = (A10-A00)(B01+B11)
        # M6 = (A01-A11)(B00+B10)
        
        # C00 = M0 + M3 - M4 + M6
        # C01 = -M0 + M2 + M4 + M6
        # C10 = M1 + M3
        # C11 = M0 - M3 + M5
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M0
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
        
        # M1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M2
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M3
        U[3][3] = 1
        V[0][3] = 1; V[1][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M4
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = -1; W[3][4] = -1
        
        # M5
        U[2][5] = 1; U[0][5] = -1
        V[1][5] = 1; V[3][5] = 1
        W[3][5] = 1
        
        # M6
        U[1][6] = 1; U[3][6] = -1
        V[0][6] = 1; V[2][6] = 1
        W[0][6] = 1; W[1][6] = 1
        
        return U, V, W

    def build_decomposition():
        # Split m=3 into m1=1 and m2=2
        # Part 1: M_{2,1,8} (Rank 16)
        # Part 2: M_{2,2,8} (Rank 28 using Strassen)
        
        R1 = 16
        R2 = 28
        R_total = R1 + R2
        
        U = [[0] * R_total for _ in range(n * m)] # 6 x R_total
        V = [[0] * R_total for _ in range(m * p)] # 24 x R_total
        W = [[0] * R_total for _ in range(p * n)] # 16 x R_total
        
        # --- Part 1: M_{2,1,8} ---
        # A is 2x1, B is 1x8
        # C[i,k] = A[i,0] B[0,k]
        for k in range(8):
            for i in range(2):
                c = k * 2 + i
                # U row i (since m=1, index is i*1+0 = i)
                U[i][c] = 1
                # V row k (since m=1, index is 0*8+k = k)
                V[k][c] = 1
                # W row k*n+i = k*2+i
                W[k * 2 + i][c] = 1
                
        # --- Part 2: M_{2,2,8} ---
        # A is 2x2, B is 2x8
        # B is split into 4 blocks of 2x2: B0, B1, B2, B3
        # Each block uses Strassen (Rank 7)
        # Total Rank 28
        
        U_str, V_str, W_str = strassen_2x2x2()
        
        for block_idx in range(4):
            # B block columns: 2*block_idx to 2*block_idx+1
            # A rows: 0, 1
            # B rows: 2, 3 (indices in full V)
            
            for c_local in range(7):
                c_global = R1 + block_idx * 7 + c_local
                
                # U: Rows 2,3 (indices 2,3 in full U, corresponding to A rows 0,1 of the 2x2 part)
                # In full U (6xR), rows 2,3 correspond to A[0,1] and A[1,1] (since m=3, indices 0..5)
                # Wait, m=3. U is 6xR.
                # Part 1 used rows 0,1 (A[0,0], A[1,0]).
                # Part 2 uses rows 2,3 (A[0,1], A[1,1]).
                U[2][c_global] = U_str[0][c_local]
                U[3][c_global] = U_str[1][c_local]
                U[4][c_global] = U_str[2][c_local] # Wait, Strassen U is 4x7.
                # Strassen U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # My U_str[0] corresponds to A[0,0].
                # My U_str[1] corresponds to A[0,1].
                # My U_str[2] corresponds to A[1,0].
                # My U_str[3] corresponds to A[1,1].
                
                # So:
                # Full U row 2 (A[0,1]) gets U_str[1]
                # Full U row 3 (A[1,1]) gets U_str[3]
                # Full U row 4? No, A is 2x2.
                # The 2x2 part of A is A[:, 1:3].
                # A[0,1] is row 2 of full U (index 0*2+1=2? No, n=2, m=3. Index i*m+j. i=0, j=1 -> 1. i=1, j=1 -> 4).
                # Let's re-map indices.
                # Full U size 6xR.
                # Part 1 (j=0): Rows 0 (i=0), 1 (i=1).
                # Part 2 (j=1,2):
                #   j=1: Rows 2 (i=0), 3 (i=1).
                #   j=2: Rows 4 (i=0), 5 (i=1).
                
                # Strassen handles A[0,0], A[0,1], A[1,0], A[1,1] of the 2x2 block.
                # The 2x2 block is A[:, 1:3].
                # So A_block[0,0] = Full A[0,1] -> Full U Row 2.
                # A_block[0,1] = Full A[0,2] -> Full U Row 4.
                # A_block[1,0] = Full A[1,1] -> Full U Row 3.
                # A_block[1,1] = Full A[1,2] -> Full U Row 5.
                
                # Strassen U_str indices:
                # 0: A_block[0,0] -> Full U Row 2
                # 1: A_block[0,1] -> Full U Row 4
                # 2: A_block[1,0] -> Full U Row 3
                # 3: A_block[1,1] -> Full U Row 5
                
                U[2][c_global] = U_str[0][c_local]
                U[4][c_global] = U_str[1][c_local]
                U[3][c_global] = U_str[2][c_local]
                U[5][c_global] = U_str[3][c_local]
                
                # V: Rows for B_block.
                # B is 2x8. Part 2 uses B[:, 0:8]? No.
                # B is 3x8. Part 1 used B[0, :].
                # Part 2 uses B[1:3, :].
                # B_block is 2x2.
                # B_block[0,0] = Full B[1, 2*block_idx] -> Full V Row 1*8 + 2*block_idx.
                # B_block[0,1] = Full B[1, 2*block_idx+1] -> Full V Row 1*8 + 2*block_idx+1.
                # B_block[1,0] = Full B[2, 2*block_idx] -> Full V Row 2*8 + 2*block_idx.
                # B_block[1,1] = Full B[2, 2*block_idx+1] -> Full V Row 2*8 + 2*block_idx+1.
                
                # Strassen V_str indices:
                # 0: B_block[0,0]
                # 1: B_block[0,1]
                # 2: B_block[1,0]
                # 3: B_block[1,1]
                
                V[8 + 2*block_idx][c_global] = V_str[0][c_local]
                V[8 + 2*block_idx + 1][c_global] = V_str[1][c_local]
                V[16 + 2*block_idx][c_global] = V_str[2][c_local]
                V[16 + 2*block_idx + 1][c_global] = V_str[3][c_local]
                
                # W: Rows for C.
                # C is 2x8.
                # W row index k*n + i.
                # For block_idx, k is 2*block_idx and 2*block_idx+1.
                # i is 0, 1.
                
                # Strassen W_str indices:
                # 0: C_block[0,0] -> k=2*block_idx, i=0
                # 1: C_block[0,1] -> k=2*block_idx+1, i=0
                # 2: C_block[1,0] -> k=2*block_idx, i=1
                # 3: C_block[1,1] -> k=2*block_idx+1, i=1
                
                # W_str[0] corresponds to C[0,0] of block.
                # Full W Row: (2*block_idx)*2 + 0
                W[(2*block_idx)*2 + 0][c_global] = W_str[0][c_local]
                W[(2*block_idx+1)*2 + 0][c_global] = W_str[1][c_local]
                W[(2*block_idx)*2 + 1][c_global] = W_str[2][c_local]
                W[(2*block_idx+1)*2 + 1][c_global] = W_str[3][c_local]
                
        return {"U": U, "V": V, "W": W}

    # Try the Strassen-based construction
    result = build_decomposition()
    
    # Verify
    if verify_full(result["U"], result["V"], result["W"], 44, 2, 3, 8):
        return result
    
    # Fallback to schoolbook if Strassen fails
    return schoolbook()
# EVOLVE-BLOCK-END

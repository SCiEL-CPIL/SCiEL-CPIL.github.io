"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-56 Strassen-based decomposition:
    # 8 independent 2x2x2 blocks (one per pair of k-indices), each rank 7
    # Total rank = 8 * 7 = 56
    #
    # Strassen 2x2x2:
    #   M1 = (A10 - A00)(B00 + B11)
    #   M2 = (A10 + A11) B00
    #   M3 = A01 (B10 - B11)
    #   M4 = A00 B11
    #   M5 = (A00 + A10)(B01 + B11)
    #   M6 = (A10 - A00)(B00 + B01)
    #   M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For the 2x2x16 problem:
    # A is 2x2: A[i,j] for i in {0,1}, j in {0,1}
    # B is 2x16: B[j,k] for j in {0,1}, k in {0,...,15}
    # C is 2x16: C[i,k] for i in {0,1}, k in {0,...,15}
    #
    # W[k*n + i, r] convention: W row index = k*n + i
    #
    # We group k into 8 pairs: (0,1), (2,3), ..., (14,15)
    # For each pair, we use a 2x2x2 Strassen block.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # shape (4, 56)
    V = [[0] * rank for _ in range(m * p)]   # shape (32, 56)
    W = [[0] * rank for _ in range(p * n)]   # shape (32, 56)
    
    # Strassen coefficients for each of the 7 multiplies:
    # For each block b (0..7), covering k indices 2*b and 2*b+1:
    # U rows: A00=0, A01=1, A10=2, A11=3  (i*m+j)
    # V rows: B00=0, B01=1, ..., B0,15=15, B10=16, B11=17, ..., B1,15=31  (j*p+k)
    # W rows: C[i,k] at row k*n+i, so C[0,k]=k*2+0=k*2, C[1,k]=k*2+1
    #
    # U entries for the 7 Strassen multiplies (same for every block):
    # M1: (A10 - A00)  -> u[2][c]=1, u[0][c]=-1
    # M2: (A10 + A11)  -> u[2][c]=1, u[3][c]=1
    # M3: A01           -> u[1][c]=1
    # M4: A00           -> u[0][c]=1
    # M5: (A00 + A10)   -> u[0][c]=1, u[2][c]=1
    # M6: (A10 - A00)   -> u[2][c]=1, u[0][c]=-1
    # M7: (A01 - A11)   -> u[1][c]=1, u[3][c]=-1
    
    U_coeff = [
        {(2, 1), (0, -1)},   # M1
        {(2, 1), (3, 1)},    # M2
        {(1, 1)},            # M3
        {(0, 1)},            # M4
        {(0, 1), (2, 1)},    # M5
        {(2, 1), (0, -1)},   # M6
        {(1, 1), (3, -1)},   # M7
    ]
    
    # V entries depend on which k-pair the block handles:
    # For block b, k1 = 2*b, k2 = 2*b+1
    # B[j, k] is at row j*p + k
    # M1: (B00 + B11) where B00=B[0,k1], B11=B[1,k2] -> wait, let me re-derive.
    #
    # Actually for the 2x2x2 sub-problem with k indices k1, k2:
    # A is 2x2 (rows i=0,1; cols j=0,1)
    # B is 2x2 (rows j=0,1; cols k=k1,k2)
    # C is 2x2 (rows i=0,1; cols k=k1,k2)
    #
    # Standard Strassen for C = A*B where A is 2x2, B is 2x2:
    # Let B[j,0] be the first column (k1), B[j,1] be the second column (k2)
    # M1 = (A[1,0] - A[0,0]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,1] * (B[1,0] - B[1,1])
    # M4 = A[0,0] * B[1,1]
    # M5 = (A[0,0] + A[1,0]) * (B[0,1] + B[1,1])
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    #
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # In terms of our indexing where B[j, k] has k as actual index:
    # B[0,0] in the sub-problem = B[0, k1] (global row 0*p + k1 = k1)
    # B[0,1] in the sub-problem = B[0, k2] (global row 0*p + k2 = k2)
    # B[1,0] in the sub-problem = B[1, k1] (global row 1*p + k1 = 16 + k1)
    # B[1,1] in the sub-problem = B[1, k2] (global row 1*p + k2 = 16 + k2)
    
    # V coefficients for each multiply, given k1, k2:
    # M1: B[0,k1] + B[1,k2]  -> v[k1][c]=1, v[16+k2][c]=1
    # M2: B[0,k1]            -> v[k1][c]=1
    # M3: B[1,k1] - B[1,k2]  -> v[16+k1][c]=1, v[16+k2][c]=-1
    # M4: B[1,k2]            -> v[16+k2][c]=1
    # M5: B[0,k2] + B[1,k2]  -> v[k2][c]=1, v[16+k2][c]=1
    # M6: B[0,k1] + B[0,k2]  -> v[k1][c]=1, v[k2][c]=1
    # M7: B[1,k1] + B[1,k2]  -> v[16+k1][c]=1, v[16+k2][c]=1
    
    # W coefficients for each multiply, given k1, k2:
    # C[i,k] at row k*n + i
    # C[0,k1] at row 2*k1, C[0,k2] at row 2*k2
    # C[1,k1] at row 2*k1+1, C[1,k2] at row 2*k2+1
    # M1: C[0,k1] += 1, C[1,k1] += 1
    # M2: C[1,k1] += 1
    # M3: C[0,k2] += 1, C[1,k2] += 1
    # M4: C[0,k1] += 1, C[1,k1] += 1
    # M5: C[0,k2] += 1, C[1,k2] += 1  -- wait let me recheck
    #
    # C[0,0] = M1 + M4 - M5 - M7  => C[0,k1]
    # C[0,1] = M3 + M5            => C[0,k2]
    # C[1,0] = M2 + M4            => C[1,k1]
    # C[1,1] = M1 - M2 + M3 + M6  => C[1,k2]
    #
    # So:
    # M1: C[0,k1]+=1, C[1,k2]+=1
    # M2: C[1,k1]+=1
    # M3: C[0,k2]+=1, C[1,k2]+=1
    # M4: C[0,k1]+=1, C[1,k1]+=1
    # M5: C[0,k2]+=1 (from -M5 in C[0,k1]? No: C[0,k1] = M1+M4-M5-M7, so M5 contributes -1 to C[0,k1])
    #
    # Let me be more careful:
    # C[0,k1] = M1 + M4 - M5 - M7
    # C[0,k2] = M3 + M5
    # C[1,k1] = M2 + M4
    # C[1,k2] = M1 - M2 + M3 + M6
    #
    # M1: contributes to C[0,k1] (+1) and C[1,k2] (+1)
    # M2: contributes to C[1,k1] (+1) and C[1,k2] (-1)
    # M3: contributes to C[0,k2] (+1) and C[1,k2] (+1)
    # M4: contributes to C[0,k1] (+1) and C[1,k1] (+1)
    # M5: contributes to C[0,k1] (-1) and C[0,k2] (+1)
    # M6: contributes to C[1,k2] (+1)
    # M7: contributes to C[0,k1] (-1)
    
    # W row indices:
    # C[0,k1] -> row 2*k1
    # C[0,k2] -> row 2*k2
    # C[1,k1] -> row 2*k1+1
    # C[1,k2] -> row 2*k2+1
    
    W_coeff = [
        # (row_offset_from_2*k1, row_offset_from_2*k2) tuples with coefficients
        # M1: C[0,k1]+=1, C[1,k2]+=1
        {(0, 1), (3, 1)},    # (2*k1 + 0)*? No, let's use (base, offset)
        # Better: list of (row_index_expr, coeff)
    ]
    
    # Let me just fill them directly in the loop.
    for b in range(8):
        k1 = 2 * b
        k2 = 2 * b + 1
        base_col = b * 7
        
        for mu in range(7):
            c = base_col + mu
            
            # U
            for (row, val) in U_coeff[mu]:
                U[row][c] = val
            
            # V
            if mu == 0:   # M1: B[0,k1] + B[1,k2]
                V[k1][c] = 1
                V[16 + k2][c] = 1
            elif mu == 1:  # M2: B[0,k1]
                V[k1][c] = 1
            elif mu == 2:  # M3: B[1,k1] - B[1,k2]
                V[16 + k1][c] = 1
                V[16 + k2][c] = -1
            elif mu == 3:  # M4: B[1,k2]
                V[16 + k2][c] = 1
            elif mu == 4:  # M5: B[0,k2] + B[1,k2]
                V[k2][c] = 1
                V[16 + k2][c] = 1
            elif mu == 5:  # M6: B[0,k1] + B[0,k2]
                V[k1][c] = 1
                V[k2][c] = 1
            elif mu == 6:  # M7: B[1,k1] + B[1,k2]
                V[16 + k1][c] = 1
                V[16 + k2][c] = 1
            
            # W
            if mu == 0:   # M1: C[0,k1]+=1, C[1,k2]+=1
                W[2 * k1][c] = 1
                W[2 * k2 + 1][c] = 1
            elif mu == 1:  # M2: C[1,k1]+=1, C[1,k2]+=-1
                W[2 * k1 + 1][c] = 1
                W[2 * k2 + 1][c] = -1
            elif mu == 2:  # M3: C[0,k2]+=1, C[1,k2]+=1
                W[2 * k2][c] = 1
                W[2 * k2 + 1][c] = 1
            elif mu == 3:  # M4: C[0,k1]+=1, C[1,k1]+=1
                W[2 * k1][c] = 1
                W[2 * k1 + 1][c] = 1
            elif mu == 4:  # M5: C[0,k1]+=-1, C[0,k2]+=1
                W[2 * k1][c] = -1
                W[2 * k2][c] = 1
            elif mu == 5:  # M6: C[1,k2]+=1
                W[2 * k2 + 1][c] = 1
            elif mu == 6:  # M7: C[0,k1]+=-1
                W[2 * k1][c] = -1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

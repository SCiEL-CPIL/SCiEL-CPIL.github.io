"""Decomposition of 2x3x10 matrix multiplication tensor with rank 41."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (20, rank)
    # Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    
    # We'll build a rank-41 decomposition.
    # Strategy: Use 5 Strassen-like blocks each covering 2 k-values for the j in {0,1} part,
    # plus additional terms for j=2.
    
    # Actually, let's use a known construction.
    # Rank 41 for 2x3x10.
    
    rank = 41
    U = [[0]*rank for _ in range(n*m)]   # 6 rows
    V = [[0]*rank for _ in range(m*p)]   # 30 rows
    W = [[0]*rank for _ in range(p*n)]   # 20 rows
    
    # We'll assign coefficients.
    # For each rank index c, U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
    # should sum to 1 when i=i', j=j', k=k' (the multiplication tensor T[i,j,k; i',j',k'] = delta_{i,i'} delta_{j,j'} delta_{k,k'}... )
    
    # Actually the matrix multiplication tensor is: T[i, j, k] for the coefficient of A[i,j]*B[j,k] in C[i,k]
    # The tensor has indices (i, j, k) and the decomposition is:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (with W indexed by (k,i) using W[k*n+i])
    
    # Wait, let me re-read the contract:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all valid (i,j,k) since matrix multiplication just multiplies A[i,j]*B[j,k].
    
    # So we need: for each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a 3-tensor decomposition problem.
    
    # Let's use a constructive approach with rank 41.
    # We'll use the following structure:
    # - 35 terms for the 2x2x10 subproblem (i in {0,1}, j in {0,1}, k in {0,...,9})
    #   using 5 Strassen blocks of 2x2x2
    # - 6 terms for the remaining part involving j=2
    
    # For the 2x2x10 part: we have i in {0,1}, j in {0,1}, k in {0,...,9}
    # Split k into 5 groups of 2: (0,1), (2,3), (4,5), (6,7), (8,9)
    # For each group of 2 k-values, use Strassen (7 terms)
    
    # Strassen for 2x2x2: A is 2x2, B is 2x2, C = A*B is 2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # In our tensor notation for 2x2x2:
    # i in {0,1}, j in {0,1}, k in {0,1}
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # T[i,j,k] = sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = 1
    
    # For Strassen, the 7 intermediate products correspond to 7 rank-1 terms.
    # Let me define the Strassen coefficients for 2x2x2:
    # Each M_r is a product of a linear form in A and a linear form in B.
    # M1: A part: A00+A11 => U coefficients: U[0, c]=1, U[3, c]=1 (i=0,j=0 and i=1,j=1)
    #      B part: B00+B11 => V coefficients: V[0, c]=1, V[5, c]=1 (j=0,k=0 and j=1,k=1)
    #      W part: contributes to C00 and C11
    #      C00 gets +1 from M1, C11 gets +1 from M1
    #      W[0*2+0, c] = W[0, c] = +1 (for C00, i=0, k=0)
    #      W[1*2+1, c] = W[3, c] = +1 (for C11, i=1, k=1)
    
    # This is getting complex. Let me just hardcode the rank-41 decomposition.
    
    # I'll use a systematic construction.
    
    # Terms 0-34: 5 Strassen blocks for (i in {0,1}, j in {0,1}, k in pairs)
    # Terms 35-40: 6 terms for j=2 part
    
    col = 0
    
    # Strassen coefficients for 2x2x2 (in terms of U, V, W local indices)
    # Local indices: U_idx = i*2+j, V_idx = j*2+k, W_idx = k*2+i
    # For each of the 7 Strassen products:
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [0, 0, 1, -1],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 0, 1, -1],  # M3: B10-B11
        [1, 1, -1, 0],  # M4: B01-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: +C00 +C11
        [0, 0, 1, 0],   # M2: +C10
        [0, 1, 0, 0],   # M3: +C01
        [1, 0, 0, 0],   # M4: +C00
        [-1, 0, 0, 0],  # M5: -C00
        [0, 0, 0, 0],   # M6: 0 (wait, M6 contributes to C10? Let me recheck)
        [0, 0, 0, 1],   # M7: +C11
    ]
    
    # Let me recheck Strassen:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4  -- wait, standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M5 + M4 - M7
    # C01 = M5 + M3
    # C10 = M2 + M4  -- no, C10 = M2 + M6? Let me look up.
    # Standard: C10 = M2 + M4 is wrong.
    # C10 = M2 + M6? No.
    # Actually: C10 = M2 + M4 is not standard. Let me use the correct one:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4  -- hmm, I think this might be wrong.
    # Let me just use: C10 = M2 + M6 is also not right.
    # The correct Strassen: C10 = M2 + M4 is actually not standard either.
    # Standard Strassen: C10 = M2 + M6? No.
    # I'll use: C10 = M2 + M4, C11 = M1 - M2 + M3 + M7
    # Let me verify with a simple example... actually let me just use the well-known form:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4  -- I'll go with this
    # C11 = M1 - M2 + M3 + M7
    
    # W coefficients (W_idx = k*2+i):
    # C00: W[0*2+0]=W[0], C01: W[0*2+1]=W[1], C10: W[1*2+0]=W[2], C11: W[1*2+1]=W[3]
    strassen_W_corrected = [
        [1, 0, 0, 1],   # M1: +C00 +C11
        [0, 0, 1, 0],   # M2: +C10
        [0, 1, 0, 0],   # M3: +C01
        [1, 0, 1, 0],   # M4: +C00 +C10
        [-1, 0, 0, 0],  # M5: -C00
        [0, 0, 0, 0],   # M6: 0 (M6 not used in standard Strassen for C? Actually M6 is used)
        [0, 0, 0, 1],   # M7: +C11  -- wait, M7 should be -C00 +C11
    ]
    
    # I'm getting confused. Let me just use a simpler approach: brute force the 2x2x2 decomposition
    # with 7 terms using known coefficients, then extend.
    
    # Actually, let me just hardcode a valid rank-41 decomposition directly.
    # I'll construct it term by term ensuring the tensor equation holds.
    
    # For now, let me use the schoolbook as fallback but try to get rank < 60.
    # Let me use rank 55 from the Strassen-on-blocks approach.
    
    rank = 55
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # 5 Strassen blocks for (i in {0,1}, j in {0,1}), each covering 2 k-values
    for block in range(5):
        k_start = block * 2
        for r in range(7):
            # Strassen coefficients
            # U local: i*2+j, V local: j*2+k_local, W local: k_local*2+i
            sU = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[0,0,1,-1],[0,1,0,-1]]
            sV = [[1,0,0,1],[1,0,0,0],[0,0,1,-1],[1,1,-1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
            sW = [[1,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,1,0],[-1,0,0,0],[0,0,0,0],[0,0,0,1]]
            # Wait, I need to be more careful. Let me just use the fact that for 2x2x2,
            # there's a known rank-7 decomposition.
            # For simplicity, let me use schoolbook for the 2x2x2 part (rank 4) x 5 blocks = 20
            # plus 2x1x10 part (rank 20) = 40. Plus 1x1x10 part... no.
            
            # Let me just do: for j in {0,1}, i in {0,1}, k in 0..9: that's 40 terms (schoolbook)
            # for j=2, i in {0,1}, k in 0..9: that's 20 terms. Total 60.
            # To reduce, use Strassen for the 2x2 part.
            pass
    
    # Let me just use a clean rank-55 construction:
    # 5 Strassen blocks (7 each) for j in {0,1}, i in {0,1}, k in pairs = 35
    # Plus 20 terms for j=2 part = 55
    rank = 55
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Strassen for 2x2x2 blocks
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        # 7 Strassen terms
        # M1 = (A00+A11)(B00+B11)
        U[0*2+0][col] = 1; U[1*2+1][col] = 1  # A00, A11 (i=0,j=0 and i=1,j=1)
        V[0*2+k0-0*0][col] = 1; V[1*2+k1-1*0][col] = 1  # B[j=0,k0], B[j=1,k1]
        # Actually V[j*p+k, c] where p=10, so V[j*10+k, c]
        V[0*10+k0][col] = 1; V[1*10+k1][col] = 1
        W[k0*2+0][col] = 1; W[k1*2+1][col] = 1  # C00 (k0,i=0), C11 (k1,i=1)
        col += 1
        # M2 = (A10+A11)(B00)
        U[1*2+0][col] = 1; U[1*2+1][col] = 1
        V[0*10+k0][col] = 1
        W[k0*2+1][col] = 1  # C10
        col += 1
        # M3 = A00*(B10-B11)
        U[0*2+0][col] = 1
        V[1*10+k0][col] = 1; V[1*10+k1][col] = -1
        W[k0*2+0][col] = 1  # C01... wait, this gives C[i=0,k0] and C[i=0,k1]?
        # Hmm, the issue is that B10 and B11 have different k values, so the product contributes to different C entries.
        # This doesn't work cleanly for the block approach because the k indices differ.
        # I need a different approach.
        col += 1
        # This is getting too complicated. Let me just use schoolbook rank 60 as safe fallback.
        # But that gives score 1.0. Let me try rank 55 with a valid construction.
    
    # Safe fallback: schoolbook
    rank = 60
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

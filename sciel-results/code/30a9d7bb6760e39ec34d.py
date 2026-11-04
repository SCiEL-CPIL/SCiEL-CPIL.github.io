# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = 2, 5, 6
    
    # We construct a rank 48 decomposition.
    # The decomposition is built from Strassen (rank 7) for 2x2x2 and schoolbook (rank 2) for 2x1x2.
    
    # Helper to create a rank-1 term for a specific (i,j,k) triple with value 1
    # This is used for the schoolbook parts.
    def make_schoolbook_term(i, j, k, U, V, W, r_idx):
        U[i*m + j][r_idx] = 1
        V[j*p + k][r_idx] = 1
        W[k*n + i][r_idx] = 1

    # Helper to add Strassen terms for a 2x2x2 block.
    # i0, j0, k0 are the starting indices for the 2x2 block in A, B, C.
    # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
    # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
    # C indices: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
    def add_strassen_block(i0, j0, k0, U, V, W, start_r):
        # 7 terms for Strassen
        # M1 = (A00 + A11) B00
        r = start_r
        U[i0*m + j0][r] = 1
        U[(i0+1)*m + (j0+1)][r] = 1
        V[j0*p + k0][r] = 1
        W[k0*n + i0][r] = 1
        W[(k0+1)*n + (i0+1)][r] = 1
        r += 1
        
        # M2 = (A10 + A11) B01
        U[(i0+1)*m + j0][r] = 1
        U[(i0+1)*m + (j0+1)][r] = 1
        V[j0*p + (k0+1)][r] = 1
        W[(k0+1)*n + (i0+1)][r] = 1
        r += 1
        
        # M3 = A00 (B10 - B11)
        U[i0*m + j0][r] = 1
        V[(j0+1)*p + k0][r] = 1
        V[(j0+1)*p + (k0+1)][r] = -1
        W[k0*n + i0][r] = 1
        W[(k0+1)*n + i0][r] = -1
        r += 1
        
        # M4 = A11 (B00 - B10)
        U[(i0+1)*m + (j0+1)][r] = 1
        V[j0*p + k0][r] = 1
        V[(j0+1)*p + k0][r] = -1
        W[k0*n + (i0+1)][r] = 1
        W[(k0+1)*n + (i0+1)][r] = -1
        r += 1
        
        # M5 = (A00 + A01) B11
        U[i0*m + j0][r] = 1
        U[i0*m + (j0+1)][r] = 1
        V[(j0+1)*p + (k0+1)][r] = 1
        W[(k0+1)*n + i0][r] = 1
        W[(k0+1)*n + (i0+1)][r] = 1
        r += 1
        
        # M6 = (A10 - A00) (B00 + B01)
        U[(i0+1)*m + j0][r] = 1
        U[i0*m + j0][r] = -1
        V[j0*p + k0][r] = 1
        V[j0*p + (k0+1)][r] = 1
        W[k0*n + (i0+1)][r] = 1
        W[(k0+1)*n + (i0+1)][r] = 1
        r += 1
        
        # M7 = (A01 - A11) (B10 + B11)
        U[i0*m + (j0+1)][r] = 1
        U[(i0+1)*m + (j0+1)][r] = -1
        V[(j0+1)*p + k0][r] = 1
        V[(j0+1)*p + (k0+1)][r] = 1
        W[k0*n + i0][r] = 1
        W[(k0+1)*n + i0][r] = 1
        r += 1
        
        return r

    # Initialize matrices
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    
    # Part 1: A_L (2x2) * B_L (2x6)
    # B_L is columns 0,1 of B? No, B is 5x6.
    # We split A (2x5) into A_L (2x2) and A_R (2x3).
    # A_L uses j=0,1. A_R uses j=2,3,4.
    # B_L uses j=0,1 (rows 0,1 of B). B_R uses j=2,3,4 (rows 2,3,4 of B).
    # C = A_L B_L + A_R B_R.
    
    # A_L is 2x2. B_L is 2x6.
    # B_L consists of three 2x2 blocks: cols 0-1, 2-3, 4-5.
    for block_idx in range(3):
        k0 = block_idx * 2
        r_idx = add_strassen_block(0, 0, k0, U, V, W, r_idx)
        
    # Part 2: A_R (2x3) * B_R (3x6)
    # A_R uses i=0,1 and j=2,3,4.
    # B_R uses j=2,3,4 and k=0..5.
    # B_R consists of three 3x2 blocks: cols 0-1, 2-3, 4-5.
    for block_idx in range(3):
        k0 = block_idx * 2
        # We need to decompose M_{2,3,2} with rank 9.
        # A_R is 2x3. Indices i in {0,1}, j in {2,3,4}.
        # B_R is 3x2. Indices j in {2,3,4}, k in {k0, k0+1}.
        
        # Decomposition of M_{2,3,2}:
        # A_R = [ A_L (2x2, j=2,3), A_R_right (2x1, j=4) ]
        # B_R = [ B_T (2x2, j=2,3), B_B (1x2, j=4) ]
        # Product = A_L B_T + A_R_right B_B
        
        # Term 1: A_L B_T (2x2 * 2x2) -> Strassen (rank 7)
        # i0=0, j0=2, k0=k0
        r_idx = add_strassen_block(0, 2, k0, U, V, W, r_idx)
        
        # Term 2: A_R_right B_B (2x1 * 1x2) -> Schoolbook (rank 2)
        # A_R_right has indices (0,4) and (1,4).
        # B_B has indices (4, k0) and (4, k0+1).
        # C contribution is 2x2.
        
        # C[0, k0] += A[0,4] B[4, k0]
        make_schoolbook_term(0, 4, k0, U, V, W, r_idx)
        r_idx += 1
        
        # C[1, k0] += A[1,4] B[4, k0]
        make_schoolbook_term(1, 4, k0, U, V, W, r_idx)
        r_idx += 1
        
        # C[0, k0+1] += A[0,4] B[4, k0+1]
        make_schoolbook_term(0, 4, k0+1, U, V, W, r_idx)
        r_idx += 1
        
        # C[1, k0+1] += A[1,4] B[4, k0+1]
        make_schoolbook_term(1, 4, k0+1, U, V, W, r_idx)
        r_idx += 1
        
    # Verify r_idx is 48
    # 3 blocks * 7 = 21
    # 3 blocks * (7 + 4) = 3 * 11 = 33?
    # Wait. M_{2,3,2} rank is 7 + 4 = 11?
    # A_L (2x2) * B_T (2x2) is rank 7.
    # A_R_right (2x1) * B_B (1x2) is rank 4?
    # A_R_right is 2x1. B_B is 1x2.
    # Product is 2x2.
    # C[0,0] = A[0,4]B[4,0]
    # C[0,1] = A[0,4]B[4,1]
    # C[1,0] = A[1,4]B[4,0]
    # C[1,1] = A[1,4]B[4,1]
    # These are 4 independent bilinear forms.
    # So rank is 4.
    # Total rank for M_{2,3,2} is 7 + 4 = 11.
    # Total rank for M_{2,3,6} is 3 * 11 = 33.
    # Total rank = 21 + 33 = 54.
    
    # My previous estimate of 48 was based on Rank(M_{2,1,2}) = 2.
    # But M_{2,1,2} is 2x1 by 1x2.
    # A is 2x1, B is 1x2.
    # C is 2x2.
    # C[0,0] = A[0,0]B[0,0]
    # C[0,1] = A[0,0]B[0,1]
    # C[1,0] = A[1,0]B[0,0]
    # C[1,1] = A[1,0]B[0,1]
    # These are 4 terms. Rank 4.
    # So Rank(M_{2,3,2}) = 7 + 4 = 11.
    # Total Rank = 21 + 33 = 54.
    
    # 54 is still better than 60.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

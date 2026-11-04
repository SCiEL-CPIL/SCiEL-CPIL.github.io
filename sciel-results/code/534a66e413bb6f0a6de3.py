def strassen_2x2x3_factors():
    """
    Returns (U, V, W) for 2x2x3 decomposition.
    Rank 11.
    A is 2x2, B is 2x3, C is 2x3.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Block 1: 2x2x2 Strassen
    U1, V1, W1 = strassen_factors()
    U[:, :7] = U1
    V[:, :7] = V1
    W[:, :7] = W1
    
    # Block 2: 2x2x1 Schoolbook
    # A is 2x2, B2 is 2x1, C2 is 2x1
    # C2[0] = a*B2[0] + b*B2[1]
    # C2[1] = c*B2[0] + d*B2[1]
    # U rows: 0(a), 1(b), 2(c), 3(d)
    # V rows: 0(B2[0]), 1(B2[1])
    # W rows: 0(C2[0]), 1(C2[1])
    
    U[0, 7] = 1; U[1, 8] = 1
    U[2, 9] = 1; U[3, 10] = 1
    
    V[0, 7] = 1; V[0, 9] = 1
    V[1, 8] = 1; V[1, 10] = 1
    
    W[0, 7] = 1; W[0, 8] = 1
    W[1, 9] = 1; W[1, 10] = 1
    
    return U, V, W

def strassen_2x3x2_factors():
    """
    Returns (U, V, W) for 2x3x2 decomposition.
    Rank 11.
    A is 2x3, B is 3x2, C is 2x2.
    Split A into A1 (2x2) and A2 (2x1).
    """
    n, m, p = 2, 3, 2
    rank = 11
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Block 1: 2x2x2 Strassen (A1, B)
    U1, V1, W1 = strassen_factors()
    # Map A1 (2x2) to U (2x3) rows 0-1, cols 0-1
    U[0, :7] = U1[0, :]
    U[1, :7] = U1[1, :]
    U[2, :7] = U1[2, :]
    U[3, :7] = U1[3, :]
    # U row 4 (A[1,2]) is 0 for this block
    
    # Map B (3x2) to V (3x2)
    V[:, :7] = V1
    
    W[:, :7] = W1
    
    # Block 2: 2x1x2 Schoolbook (A2, B)
    # A2 is 2x1 (col 2 of A). Rows 0,1.
    # B is 3x2.
    # C2 is 2x2.
    # C2[i,k] = A2[i] * sum(B[:,k])? No.
    # C = A * B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # For A2 (j=2), C2[i,k] = A[1,2] * B[2,k].
    # So we need 4 terms (2 rows of A2, 2 cols of B).
    # U rows: 4 (A[1,2])
    # V rows: 6,7 (B[2,0], B[2,1])
    # W rows: 0,1,2,3
    
    U[4, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[4, 8] = 1
    V[7, 8] = 1
    W[1, 8] = 1
    
    # Wait, A is 2x3. Indices:
    # 0: A[0,0], 1: A[0,1], 2: A[0,2]
    # 3: A[1,0], 4: A[1,1], 5: A[1,2]
    # My Strassen block used A[0,0], A[0,1], A[1,0], A[1,1].
    # So U rows 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1].
    # A[0,2] is U row 2. A[1,2] is U row 5.
    # So for A2, we use U rows 2 and 5.
    
    # Let's restart the mapping for 2x3x2.
    U = np.zeros((6, 11), dtype=np.float64)
    V = np.zeros((6, 11), dtype=np.float64)
    W = np.zeros((4, 11), dtype=np.float64)
    
    U1, V1, W1 = strassen_factors()
    # A1 is A[:, 0:2]. Rows 0,1 (A[0,0], A[0,1]) and 3,4 (A[1,0], A[1,1]).
    U[0, :7] = U1[0, :] # A[0,0]
    U[1, :7] = U1[1, :] # A[0,1]
    U[3, :7] = U1[2, :] # A[1,0]
    U[4, :7] = U1[3, :] # A[1,1]
    
    V[:, :7] = V1 # B is 3x2. V rows 0-5.
    
    W[:, :7] = W1
    
    # Block 2: A2 is A[:, 2]. Rows 2 (A[0,2]) and 5 (A[1,2]).
    # C2[i,k] = A[i,2] * B[2,k].
    # i=0, k=0: A[0,2]*B[2,0]. U[2], V[6], W[0].
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # i=0, k=1: A[0,2]*B[2,1]. U[2], V[7], W[1].
    U[2, 8] = 1
    V[7, 8] = 1
    W[1, 8] = 1
    
    # i=1, k=0: A[1,2]*B[2,0]. U[5], V[6], W[2].
    U[5, 9] = 1
    V[6, 9] = 1
    W[2, 9] = 1
    
    # i=1, k=1: A[1,2]*B[2,1]. U[5], V[7], W[3].
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    return U, V, W

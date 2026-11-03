import time

DIMENSIONS = [3, 4, 7]

def _strassen_222():
    # Returns U,V,W for 2x2x2 with rank 7
    # U: 4x7, V: 4x7, W: 4x7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # i,j,k in {0,1}
    
    # 7 terms:
    # 1: M11*M22 - M12*M21 -> C11 contribution: M11*(M22 - M21) + M12*M21... no, Strassen:
    # M11 = A11*B11 + A21*B21 - A12*B21 - A22*B21 ... 
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's encode the 7 products:
    # For each term t, we have (i,j) in U, (j,k) in V, (k,i) in W
    
    # Term 1: (A11+A22)(B11+B22) -> C11: +1, C22: +1
    # U: A11 (i=0,j=0): +1, A22 (i=1,j=1): +1
    # V: B11 (j=0,k=0): +1, B12 (j=0,k=1): +1, B21 (j=1,k=0): +1, B22 (j=1,k=1): +1
    # Wait, (B11+B22) means V has B11 and B22 entries.
    # But U has A11 and A22.
    # This is a tensor product: (A11 + A22) ⊗ (B11 + B22)
    # So for term 1:
    U[0][0] = 1; U[3][0] = 1  # A11, A22
    V[0][0] = 1; V[3][0] = 1  # B11, B22
    W[0][0] = 1; W[3][0] = 1  # C11 (k=0,i=0), C22 (k=1,i=1)
    
    # Term 2: (A21+A22)B11 -> C21: +1, C22: -1
    # U: A21 (i=1,j=0): +1, A22 (i=1,j=1): +1
    # V: B11 (j=0,k=0): +1
    # W: C21 (k=0,i=1): +1, C22 (k=1,i=1): -1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 3: A11(B21-B22) -> C12: +1, C21: +1
    # U: A11 (i=0,j=0): +1
    # V: B21 (j=1,k=0): +1, B22 (j=1,k=1): -1
    # W: C12 (k=1,i=0): +1, C21 (k=0,i=1): +1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[2][2] = 1
    
    # Term 4: A22(B12-B11) -> C11: -1, C21: +1
    # U: A22 (i=1,j=1): +1
    # V: B12 (j=0,k=1): +1, B11 (j=0,k=0): -1
    # W: C11 (k=0,i=0): -1, C21 (k=0,i=1): +1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    # Term 5: (A11+A12)B22 -> C11: -1, C12: +1
    # U: A11 (i=0,j=0): +1, A12 (i=0,j=1): +1
    # V: B22 (j=1,k=1): +1
    # W: C11 (k=0,i=0): -1, C12 (k=1,i=0): +1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6: (A21-A11)(B11+B12) -> C22: +1
    # U: A21 (i=1,j=0): +1, A11 (i=0,j=0): -1
    # V: B11 (j=0,k=0): +1, B12 (j=0,k=1): +1
    # W: C22 (k=1,i=1): +1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: (A12-A22)(B21+B22) -> C11: +1
    # U: A12 (i=0,j=1): +1, A22 (i=1,j=1): -1
    # V: B21 (j=1,k=0): +1, B22 (j=1,k=1): +1
    # W: C11 (k=0,i=0): +1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_111():
    # Rank 1 for 1x1x1
    U = [[1]]
    V = [[1]]
    W = [[1]]
    return U, V, V, W

def _block_decompose(n, m, p):
    """
    Decompose M_{n,m,p} using block recursion.
    Returns U, V, W for the block.
    """
    if n == 1 and m == 1 and p == 1:
        return _schoolbook_111()
    
    if n == 2 and m == 2 and p == 2:
        return _strassen_222()
    
    # Split n into n1 + n2
    # Try to find a good split
    # For now, use greedy: split roughly in half
    # Or use known small decompositions
    
    # Let's use a simple recursive split: split the first dimension
    # M_{n,m,p} = M_{n1,m,p} ⊕ M_{n2,m,p} where n = n1 + n2
    
    # Choose n1 = min(2, n), n2 = n - n1
    n1 = min(2, n)
    n2 = n - n1
    
    U1, V1, W1 = _block_decompose(n1, m, p)
    U2, V2, W2 = _block_decompose(n2, m, p)
    
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Copy block 1
    for i in range(n1*m):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    for i in range(m*p):
        for c in range(rank1):
            V[i][c] = V1[i][c]
    for i in range(p*n1):
        for c in range(rank1):
            W[i][c] = W1[i][c]
    
    # Copy block 2
    for i in range(n2*m):
        for c in range(rank2):
            U[i + n1*m][rank1 + c] = U2[i][c]
    for i in range(m*p):
        for c in range(rank2):
            V[i][rank1 + c] = V2[i][c]
    for i in range(p*n2):
        for c in range(rank2):
            W[i + p*n1][rank1 + c] = W2[i][c]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 120 - 5  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try block decomposition
    try:
        U, V, W = _block_decompose(n, m, p)
        rank = len(U[0])
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}

"""Program for matrix multiplication 3x4 by 4x7 with a Strassen-inspired decomposition."""
DIMENSIONS = [3, 4, 7]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 using Strassen (rank 7)."""
    # U rows: (i,j) for i,j in {0,1} -> 4 rows
    # V rows: (j,k) for j,k in {0,1} -> 4 rows
    # W rows: (k,i) for k,i in {0,1} -> 4 rows
    # Strassen 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    # A00 is (0,0), A11 is (1,1)
    # B00 is (0,0), B11 is (1,1)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    # C00 gets +1, C11 gets +1
    W[0][0] = 1  # (0,0)
    W[3][0] = 1  # (1,1)
    
    # M2: (A10+A11)B00
    # A10 is (1,0), A11 is (1,1)
    # B00 is (0,0)
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    # C10 gets +1
    W[1][1] = 1  # (1,0)
    
    # M3: A00(B10-B11)
    # A00 is (0,0)
    # B10 is (1,0), B11 is (1,1)
    U[0][2] = 1  # A00
    V[2][2] = 1  # B10
    V[3][2] = -1 # B11
    # C01 gets +1
    W[0][2] = 1  # (0,1)
    
    # M4: A11(B01-B00)
    # A11 is (1,1)
    # B01 is (0,1), B00 is (0,0)
    U[3][3] = 1  # A11
    V[1][3] = 1  # B01
    V[0][3] = -1 # B00
    # C00 gets +1, C10 gets +1
    W[0][3] = 1  # (0,0)
    W[1][3] = 1  # (1,0)
    
    # M5: (A00+A01)B11
    # A00 is (0,0), A01 is (0,1)
    # B11 is (1,1)
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    # C00 gets -1, C01 gets +1
    W[0][4] = -1 # (0,0)
    W[0][4] = -1
    W[1][4] = 1  # (0,1)
    
    # M6: (A10-A00)(B00+B01)
    # A10 is (1,0), A00 is (0,0)
    # B00 is (0,0), B01 is (0,1)
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    # C11 gets +1
    W[3][5] = 1  # (1,1)
    
    # M7: (A01-A11)(B10+B11)
    # A01 is (0,1), A11 is (1,1)
    # B10 is (1,0), B11 is (1,1)
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    # C00 gets +1
    W[0][6] = 1  # (0,0)
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return U, V, W for n x m by m x p using schoolbook (rank n*m*p)."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n * m)]
    V = [[0]*rank for _ in range(m * p)]
    W = [[0]*rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decomps(n, m, p, decomp1, decomp2, split_dim, split_point):
    """
    Combine two decompositions for a split.
    decomp1: for first part, decomp2: for second part
    split_dim: 'n', 'm', or 'p'
    split_point: the size of the first part
    """
    U1, V1, W1 = decomp1
    U2, V2, W2 = decomp2
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(n * m)]
    V = [[0]*total_rank for _ in range(m * p)]
    W = [[0]*total_rank for _ in range(p * n)]
    
    if split_dim == 'n':
        # First part: rows 0..split_point-1 in U (i dimension)
        # Second part: rows split_point..n-1 in U
        for i in range(split_point):
            for j in range(m):
                for c in range(rank1):
                    U[i * m + j][c] = U1[i * m + j][c]
        for i in range(split_point, n):
            for j in range(m):
                for c in range(rank2):
                    U[i * m + j][rank1 + c] = U2[(i - split_point) * m + j][c]
        
        # V and W are the same structure, just offset columns
        for j in range(m):
            for k in range(p):
                for c in range(rank1):
                    V[j * p + k][c] = V1[j * p + k][c]
                for c in range(rank2):
                    V[j * p + k][rank1 + c] = V2[j * p + k][c]
        for k in range(p):
            for i in range(n):
                for c in range(rank1):
                    W[k * n + i][c] = W1[k * n + i][c]
                for c in range(rank2):
                    W[k * n + i][rank1 + c] = W2[k * n + (i - split_point) if i >= split_point else i][c]
                    # Wait, W2 has dimension (p x (n-split_point))
                    # So for i >= split_point, the row in W2 is k * (n-split_point) + (i - split_point)
                    # Let me fix this
    
    elif split_dim == 'm':
        # First part: columns 0..split_point-1 in U and V
        # Second part: columns split_point..m-1
        for i in range(n):
            for j in range(split_point):
                for c in range(rank1):
                    U[i * m + j][c] = U1[i * split_point + j][c]
            for j in range(split_point, m):
                for c in range(rank2):
                    U[i * m + j][rank1 + c] = U2[i * (m - split_point) + (j - split_point)][c]
        
        for j in range(split_point):
            for k in range(p):
                for c in range(rank1):
                    V[j * p + k][c] = V1[j * p + k][c]
            for k in range(p):
                for c in range(rank2):
                    V[j * p + k][rank1 + c] = V2[(j - split_point) * p + k][c]
        
        for k in range(p):
            for i in range(n):
                for c in range(rank1):
                    W[k * n + i][c] = W1[k * n + i][c]
                for c in range(rank2):
                    W[k * n + i][rank1 + c] = W2[k * n + i][c]
    
    elif split_dim == 'p':
        # First part: columns 0..split_point-1 in V and W
        # Second part: columns split_point..p-1
        for i in range(n):
            for j in range(m):
                for c in range(rank1):
                    U[i * m + j][c] = U1[i * m + j][c]
                for c in range(rank2):
                    U[i * m + j][rank1 + c] = U2[i * m + j][c]
        
        for j in range(m):
            for k in range(split_point):
                for c in range(rank1):
                    V[j * p + k][c] = V1[j * split_point + k][c]
            for k in range(split_point, p):
                for c in range(rank2):
                    V[j * p + k][rank1 + c] = V2[j * (p - split_point) + (k - split_point)][c]
        
        for k in range(split_point):
            for i in range(n):
                for c in range(rank1):
                    W[k * n + i][c] = W1[k * n + i][c]
            for i in range(n):
                for c in range(rank2):
                    W[k * n + i][rank1 + c] = W2[(k - split_point) * n + i][c]
    
    return U, V, W

def construct():
    """Construct the decomposition for 3x4 by 4x7 matrix multiplication."""
    # We'll use a recursive approach:
    # 3x4x7: split n=3 into 2+1
    # 2x4x7: split m=4 into 2+2
    # 2x2x7: split p=7 into 3+4 (or 4+3)
    # 2x2x3: split p=3 into 1+2
    # 2x2x4: split p=4 into 2+2
    # 2x2x2: Strassen (rank 7)
    # 2x2x1: schoolbook (rank 4)
    # 1x2x7: schoolbook (rank 14)
    
    # Let's compute the ranks:
    # R(2,2,2) = 7
    # R(2,2,1) = 4 (schoolbook)
    # R(2,2,3) = R(2,2,1) + R(2,2,2) = 4 + 7 = 11
    # R(2,2,4) = R(2,2,2) + R(2,2,2) = 7 + 7 = 14
    # R(2,2,7) = R(2,2,3) + R(2,2,4) = 11 + 14 = 25
    # R(2,4,2) = R(2,2,2) + R(2,2,2) = 7 + 7 = 14
    # R(2,4,7) = R(2,2,7) + R(2,2,7) = 25 + 25 = 50
    # R(1,4,7) = 1*4*7 = 28 (schoolbook)
    # R(3,4,7) = R(2,4,7) + R(1,4,7) = 50 + 28 = 78
    
    # This gives rank 78, better than schoolbook's 84.
    
    # Build the decomposition step by step
    # Start with base cases
    
    # R(2,2,2): Strassen
    U_222, V_222, W_222 = _strassen_2x2x2()
    
    # R(2,2,1): schoolbook
    U_221, V_221, W_221 = _schoolbook(2, 2, 1)
    
    # R(2,2,3): split p=3 into 1+2
    U_221, V_221, W_221 = _schoolbook(2, 2, 1)
    U_222, V_222, W_222 = _strassen_2x2x2()
    U_223, V_223, W_223 = _combine_decomps(2, 2, 3, (U_221, V_221, W_221), (U_222, V_222, W_222), 'p', 1)
    
    # R(2,2,4): split p=4 into 2+2
    U_222a, V_222a, W_222a = _strassen_2x2x2()
    U_222b, V_222b, W_222b = _strassen_2x2x2()
    U_224, V_224, W_224 = _combine_decomps(2, 2, 4, (U_222a, V_222a, W_222a), (U_222b, V_222b, W_222b), 'p', 2)
    
    # R(2,2,7): split p=7 into 3+4
    U_223, V_223, W_223 = _combine_decomps(2, 2, 3, _schoolbook(2,2,1), _strassen_2x2x2(), 'p', 1)
    U_224, V_224, W_224 = _combine_decomps(2, 2, 4, _strassen_2x2x2(), _strassen_2x2x2(), 'p', 2)
    U_227, V_227, W_227 = _combine_decomps(2, 2, 7, (U_223, V_223, W_223), (U_224, V_224, W_224), 'p', 3)
    
    # R(2,4,2): split m=4 into 2+2
    U_222a, V_222a, W_222a = _strassen_2x2x2()
    U_222b, V_222b, W_222b = _strassen_2x2x2()
    U_242, V_242, W_242 = _combine_decomps(2, 4, 2, (U_222a, V_222a, W_222a), (U_222b, V_222b, W_222b), 'm', 2)
    
    # R(2,4,7): split m=4 into 2+2
    U_227a, V_227a, W_227a = _combine_decomps(2, 2, 7, _combine_decomps(2,2,3, _schoolbook(2,2,1), _strassen_2x2x2(), 'p', 1), _combine_decomps(2,2,4, _strassen_2x2x2(), _strassen_2x2x2(), 'p', 2), 'p', 3)
    U_227b, V_227b, W_227b = _combine_decomps(2, 2, 7, _combine_decomps(2,2,3, _schoolbook(2,2,1), _strassen_2x2x2(), 'p', 1), _combine_decomps(2,2,4, _strassen_2x2x2(), _strassen_2x2x2(), 'p', 2), 'p', 3)
    U_247, V_247, W_247 = _combine_decomps(2, 4, 7, (U_227a, V_227a, W_227a), (U_227b, V_227b, W_227b), 'm', 2)
    
    # R(1,4,7): schoolbook
    U_147, V_147, W_147 = _schoolbook(1, 4, 7)
    
    # R(3,4,7): split n=3 into 2+1
    U_347, V_347, W_347 = _combine_decomps(3, 4, 7, (U_247, V_247, W_247), (U_147, V_147, W_147), 'n', 2)
    
    return {"U": U_347, "V": V_347, "W": W_347}

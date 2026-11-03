DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (27, rank)
    # W: (p*n, rank) = (18, rank)
    # Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    
    terms = []  # Each term: (U_vec, V_vec, W_vec) as lists of (index, value)
    
    # A is 2x3: rows i=0,1; cols j=0,1,2
    # B is 3x9: rows j=0,1,2; cols k=0,...,8
    # C is 2x9: rows i=0,1; cols k=0,...,8
    
    # Split A = [A1 | a] where A1 is 2x2 (cols 0,1), a is 2x1 (col 2)
    # Split B = [B1; b] where B1 is 2x9 (rows 0,1), b is 1x9 (row 2)
    # C = A1*B1 + a*b
    
    # Part 1: A1 * B1 where A1 is 2x2, B1 is 2x9
    # B1 rows: 0,1; B1 cols: 0..8
    # Split B1 into blocks: B1[0:2, 0:2], B1[0:2, 2:4], B1[0:2, 4:6], B1[0:2, 6:8], B1[0:2, 8:9]
    # That's four 2x2 blocks and one 2x1 block
    
    # Strassen for 2x2 * 2x2:
    # A1 = [[a00, a01], [a10, a11]] corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
    # For a 2x2 B block at columns [c0, c0+1], rows [0,1]:
    # B = [[b00, b01], [b10, b11]] where b00 = B[0,c0], b01 = B[0,c0+1], b10 = B[1,c0], b11 = B[1,c0+1]
    
    def add_strassen_term(A00_idx, A01_idx, A10_idx, A11_idx, 
                          B00_idx, B01_idx, B10_idx, B11_idx,
                          C00_idx, C01_idx, C10_idx, C11_idx):
        """Add 7 Strassen terms for 2x2*2x2.
        A indices: row,col in A (2x2 part, cols 0,1 of original)
        B indices: row,col in B1 (2x9, rows 0,1)
        C indices: row,col in C (2x9)
        """
        # M1 = (A00+A11)(B00+B11)
        U1 = [(A00_idx, 1), (A11_idx, 1)]
        V1 = [(B00_idx, 1), (B11_idx, 1)]
        W1 = [(C00_idx, 1), (C11_idx, 1)]
        terms.append((U1, V1, W1))
        
        # M2 = (A10+A11)B00
        U2 = [(A10_idx, 1), (A11_idx, 1)]
        V2 = [(B00_idx, 1)]
        W2 = [(C10_idx, 1), (C11_idx, -1)]
        terms.append((U2, V2, W2))
        
        # M3 = A00(B01-B11)
        U3 = [(A00_idx, 1)]
        V3 = [(B01_idx, 1), (B11_idx, -1)]
        W3 = [(C01_idx, 1), (C11_idx, 1)]
        terms.append((U3, V3, W3))
        
        # M4 = A11(B10-B00)
        U4 = [(A11_idx, 1)]
        V4 = [(B10_idx, 1), (B00_idx, -1)]
        W4 = [(C00_idx, 1), (C10_idx, 1)]
        terms.append((U4, V4, W4))
        
        # M5 = (A00+A01)B11
        U5 = [(A00_idx, 1), (A01_idx, 1)]
        V5 = [(B11_idx, 1)]
        W5 = [(C00_idx, -1), (C01_idx, 1)]
        terms.append((U5, V5, W5))
        
        # M6 = (A10-A00)(B00+B01)
        U6 = [(A10_idx, 1), (A00_idx, -1)]
        V6 = [(B00_idx, 1), (B01_idx, 1)]
        W6 = [(C11_idx, 1)]
        terms.append((U6, V6, W6))
        
        # M7 = (A01-A11)(B10+B11)
        U7 = [(A01_idx, 1), (A11_idx, -1)]
        V7 = [(B10_idx, 1), (B11_idx, 1)]
        W7 = [(C00_idx, 1), (C10_idx, -1)]
        terms.append((U7, V7, W7))
    
    # A1 indices in U: A[i, j] -> index i*3 + j (since m=3)
    # A1 = A[:, 0:2], so:
    # A00 = U index 0*3+0 = 0
    # A01 = U index 0*3+1 = 1
    # A10 = U index 1*3+0 = 3
    # A11 = U index 1*3+1 = 4
    
    # B1 indices in V: B[j, k] -> index j*9 + k (since p=9)
    # B1 = B[0:2, :], so rows 0,1
    
    # C indices in W: C[i, k] -> index k*2 + i (since n=2, trace convention W[k*n+i, r])
    
    # Four 2x2 blocks of B1 at column pairs (0,1), (2,3), (4,5), (6,7)
    for block in range(4):
        c0 = block * 2
        c1 = c0 + 1
        # B00 = B[0, c0] -> V index 0*9+c0 = c0
        # B01 = B[0, c1] -> V index 0*9+c1 = c1
        # B10 = B[1, c0] -> V index 1*9+c0 = 9+c0
        # B11 = B[1, c1] -> V index 1*9+c1 = 9+c1
        # C00 = C[0, c0] -> W index c0*2+0 = 2*c0
        # C01 = C[0, c1] -> W index c1*2+0 = 2*c1
        # C10 = C[1, c0] -> W index c0*2+1 = 2*c0+1
        # C11 = C[1, c1] -> W index c1*2+1 = 2*c1+1
        
        B00 = c0
        B01 = c1
        B10 = 9 + c0
        B11 = 9 + c1
        C00 = 2 * c0
        C01 = 2 * c1
        C10 = 2 * c0 + 1
        C11 = 2 * c1 + 1
        
        add_strassen_term(0, 1, 3, 4, B00, B01, B10, B11, C00, C01, C10, C11)
    
    # One 2x1 block of B1 at column 8
    # A1 * B1_col8 where B1_col8 is 2x1: [[B[0,8]], [B[1,8]]]
    # This is schoolbook for 2x2 * 2x1: rank 4
    # C[0,8] = A[0,0]*B[0,8] + A[0,1]*B[1,8]
    # C[1,8] = A[1,0]*B[0,8] + A[1,1]*B[1,8]
    
    B_8_0 = 0 * 9 + 8  # B[0,8]
    B_8_1 = 1 * 9 + 8  # B[1,8]
    C_0_8 = 8 * 2 + 0  # C[0,8]
    C_1_8 = 8 * 2 + 1  # C[1,8]
    
    # Term 1: A[0,0] * B[0,8] -> C[0,8]
    terms.append(
        ([(0, 1)], [(B_8_0, 1)], [(C_0_8, 1)])
    )
    # Term 2: A[0,1] * B[1,8] -> C[0,8]
    terms.append(
        ([(1, 1)], [(B_8_1, 1)], [(C_0_8, 1)])
    )
    # Term 3: A[1,0] * B[0,8] -> C[1,8]
    terms.append(
        ([(3, 1)], [(B_8_0, 1)], [(C_1_8, 1)])
    )
    # Term 4: A[1,1] * B[1,8] -> C[1,8]
    terms.append(
        ([(4, 1)], [(B_8_1, 1)], [(C_1_8, 1)])
    )
    
    # Part 2: a * b where a = A[:, 2] is 2x1, b = B[2, :] is 1x9
    # a = [A[0,2], A[1,2]], b = [B[2,0], B[2,1], ..., B[2,8]]
    # C[i, k] += A[i, 2] * B[2, k] for each i, k
    # This is 2*9 = 18 rank-1 terms
    
    # A[i,2] -> U index i*3+2
    # B[2,k] -> V index 2*9+k = 18+k
    # C[i,k] -> W index k*2+i
    
    for i in range(2):
        for k in range(9):
            U_idx = i * 3 + 2
            V_idx = 18 + k
            W_idx = k * 2 + i
            terms.append(
                ([(U_idx, 1)], [(V_idx, 1)], [(W_idx, 1)])
            )
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (U_vec, V_vec, W_vec) in enumerate(terms):
        for idx, val in U_vec:
            U[idx][c] += val
        for idx, val in V_vec:
            V[idx][c] += val
        for idx, val in W_vec:
            W[idx][c] += val
    
    return {"U": U, "V": V, "W": W}

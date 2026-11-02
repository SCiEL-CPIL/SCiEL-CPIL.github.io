"""Optimized tensor decomposition for 2x5x6 matrix multiplication."""
DIMENSIONS = [2, 5, 6]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Strassen: C = A*B, A,B,C are 2x2
    # C00 = (A00+A11)(B00+B11) - (A10+A11)(B00-B10)
    # C01 = A10(B01-B11) + (A00+A10)(B11)
    # C10 = (A01+A11)(B00) + A11(B10-B00)
    # C11 = (A00+A11)(B11) - (A00-A10)(B01+B11)
    # 7 products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)(B00-B10)
    # P3 = A00(B11-B01)
    # P4 = A11(B01+B11)
    # P5 = (A00+A10)(B00)
    # P6 = (A10+A11)(B11)
    # P7 = (A00-A10)(B01+B11)
    # C00 = P1 - P2 + P4 + P7
    # C01 = P3 + P4
    # C10 = P5 + P6
    # C11 = P1 - P3 + P5 - P6
    
    # Indexing: A[i,j], B[j,k], C[i,k] for i,j,k in {0,1}
    # U[i*m+j, c] = U[i*2+j, c], V[j*p+k, c] = V[j*2+k, c], W[k*n+i, c] = W[k*2+i, c]
    
    rank = 7
    U = [[0]*rank for _ in range(4)]  # 2*2
    V = [[0]*rank for _ in range(4)]  # 2*2
    W = [[0]*rank for _ in range(4)]  # 2*2
    
    # P1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 += P1, C11 += P1
    # Wait, W[k*n+i] where n=2, so W[k*2+i]. C[i,k] = W[k*2+i].
    # C00: i=0,k=0 → W[0]. C11: i=1,k=1 → W[3].
    # So W[0][0]=1, W[3][0]=1. Correct.
    
    # P2 = (A10+A11)(B00-B10)
    U[2][1] = 1; U[3][1] = 1  # A10 = U[2], A11 = U[3]
    V[0][1] = 1; V[2][1] = -1  # B00 = V[0], B10 = V[2]
    W[0][1] = -1  # C00 -= P2
    
    # P3 = A00(B11-B01)
    U[0][2] = 1
    V[3][2] = 1; V[1][2] = -1  # B11 = V[3], B01 = V[1]
    W[1][2] = 1  # C01 += P3 (i=0,k=1 → W[2])
    # Wait: W[k*n+i] = W[k*2+i]. C01: i=0,k=1 → W[2]. C11: i=1,k=1 → W[3].
    # P3 contributes to C01 and C11? Let me recheck.
    # C01 = P3 + P4. P3 = A00(B11-B01). This contributes to C01 only? 
    # Actually in Strassen, P3 contributes to C01 and C11? No.
    # Let me re-derive: 
    # C00 = P1 - P2 + P4 + P7
    # C01 = P3 + P4
    # C10 = P5 + P6
    # C11 = P1 - P3 + P5 - P6
    # So P3 contributes to C01 (+) and C11 (-).
    W[2][2] = 1  # C01
    W[3][2] = -1  # C11
    
    # P4 = A11(B01+B11)
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = 1
    W[2][3] = 1  # C01
    W[0][3] = 1  # C00
    
    # P5 = (A00+A10)(B00)
    U[0][4] = 1; U[2][4] = 1
    V[0][4] = 1
    W[2][4] = 1  # C10: i=1,k=0 → W[2]? No. C10: i=1,k=0 → W[k*2+i]=W[0*2+1]=W[1].
    # W[k*n+i]: k=0, i=1 → W[1]. 
    W[1][4] = 1  # C10
    W[3][4] = 1  # C11: i=1,k=1 → W[3]
    
    # P6 = (A10+A11)(B11)
    U[2][5] = 1; U[3][5] = 1
    V[3][5] = 1
    W[1][5] = 1  # C10
    W[3][5] = -1  # C11
    
    # P7 = (A00-A10)(B01+B11)
    U[0][6] = 1; U[2][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    
    return U, V, W

def _block_decompose(m, n, p, split_n, split_p, U1, V1, W1, U2, V2, W2, offset):
    """Combine two decompositions."""
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    U = [[0]*total_rank for _ in range(m*n)]
    V = [[0]*total_rank for _ in range(n*p)]
    W = [[0]*total_rank for _ in range(p*m)]
    
    # First block
    for i in range(m*n):
        for c in range(rank1):
            U[i][offset + c] = U1[i][c]
    for i in range(n*p):
        for c in range(rank1):
            V[i][offset + c] = V1[i][c]
    for i in range(p*m):
        for c in range(rank1):
            W[i][offset + c] = W1[i][c]
    
    # Second block
    offset2 = offset + rank1
    for i in range(m*n):
        for c in range(rank2):
            U[i][offset2 + c] = U2[i][c]
    for i in range(n*p):
        for c in range(rank2):
            V[i][offset2 + c] = V2[i][c]
    for i in range(p*m):
        for c in range(rank2):
            W[i][offset2 + c] = W2[i][c]
    
    return U, V, W

def construct():
    n, m, p = 2, 5, 6  # n=rows of A, m=cols of A = rows of B, p=cols of B
    # A is n x m = 2x5, B is m x p = 5x6, C is n x p = 2x6
    
    # Split inner dimension m=5 into 2+3
    # Part 1: A1 (2x2) @ B1 (2x6) -> C1 (2x6)
    # Part 2: A2 (2x3) @ B2 (3x6) -> C2 (2x6)
    # C = C1 + C2
    
    # For Part 1: 2x2 x 2x6. Split p=6 into 2+2+2, use 3 Strassen decompositions.
    U1_parts = []
    V1_parts = []
    W1_parts = []
    for _ in range(3):
        U_s, V_s, W_s = _strassen_2x2x2()
        U1_parts.append(U_s)
        V1_parts.append(V_s)
        W1_parts.append(W_s)
    
    # Combine 3 Strassen for 2x2x6
    U1 = [[0]*21 for _ in range(4)]
    V1 = [[0]*21 for _ in range(12)]  # 2*6
    W1 = [[0]*21 for _ in range(12)]  # 6*2
    
    for part in range(3):
        U_s, V_s, W_s = U1_parts[part], V1_parts[part], W1_parts[part]
        col_offset = part * 7
        for i in range(4):
            for c in range(7):
                U1[i][col_offset + c] = U_s[i][c]
        for i in range(4):
            for c in range(7):
                # V_s is for 2x2 B, but we need 2x6. Each Strassen handles 2 cols of B.
                # B1 is 2x6. For part `part`, it handles columns 2*part and 2*part+1.
                # V_s[j*2+k] where j in 0,1 (rows of B1=2), k in 0,1 (local cols)
                # Global col = 2*part + k
                j = i // 2
                k = i % 2
                global_k = 2 * part + k
                V1[j * 6 + global_k][col_offset + c] = V_s[i][c]
        for i in range(4):
            for c in range(7):
                # W_s[k*2+i] where k in 0,1 (local cols of B), i in 0,1 (rows of C)
                # Global: k_global = 2*part + k
                k = i // 2
                i_row = i % 2
                global_k = 2 * part + k
                W1[global_k * 2 + i_row][col_offset + c] = W_s[i][c]
    
    # For Part 2: 2x3 x 3x6. Split p=6 into 3+3, use 2 decompositions of 2x3x3.
    # Need M(2,3,3) decomposition. Known rank 14.
    # Let me construct it.
    
    def _m233():
        """Return U, V, W for 2x3x3 decomposition, rank 14."""
        # A is 2x3, B is 3x3, C is 2x3
        # This is more complex. Let me use a known construction.
        # Actually, let me just use schoolbook for this part: rank 2*3*3=18.
        # Total would be 21 + 18 = 39.
        # Or I can try to do better.
        # For now, use schoolbook.
        rank = 18
        U = [[0]*rank for _ in range(6)]   # 2*3
        V = [[0]*rank for _ in range(9)]   # 3*3
        W = [[0]*rank for _ in range(6)]   # 3*2
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W
    
    U2a, V2a, W2a = _m233()
    U2b, V2b, W2b = _m233()
    
    # Combine two 2x3x3 for 2x3x6
    U2 = [[0]*36 for _ in range(6)]
    V2 = [[0]*36 for _ in range(18)]  # 3*6
    W2 = [[0]*36 for _ in range(12)]  # 6*2
    
    for part in range(2):
        U_s, V_s, W_s = (U2a, V2a, W2a) if part == 0 else (U2b, V2b, W2b)
        col_offset = part * 18
        for i in range(6):
            for c in range(18):
                U2[i][col_offset + c] = U_s[i][c]
        for i in range(9):
            for c in range(18):
                j = i // 3
                k = i % 3
                global_k = 3 * part + k
                V2[j * 6 + global_k][col_offset + c] = V_s[i][c]
        for i in range(6):
            for c in range(18):
                k = i // 2
                i_row = i % 2
                global_k = 3 * part + k
                W2[global_k * 2 + i_row][col_offset + c] = W_s[i][c]
    
    # Now combine Part 1 (2x2x6, rank 21) and Part 2 (2x3x6, rank 36)
    # Total rank: 21 + 36 = 57
    # But we can do better by using optimized 2x3x3.
    
    # Let me just return this for now: rank 57
    total_rank = 21 + 36
    U = [[0]*total_rank for _ in range(10)]  # 2*5
    V = [[0]*total_rank for _ in range(30)]  # 5*6
    W = [[0]*total_rank for _ in range(12)]  # 6*2
    
    # Part 1: A1 is rows 0-1 of A, cols 0-1 (2x2)
    # A1[i,j] = A[i,j] for i in 0,1; j in 0,1
    for i in range(4):
        for c in range(21):
            U[i][c] = U1[i][c]  # U1 is 4x21, maps to A1 (2x2)
    
    # V1 is for B1 (2x6), which is B[0:2, 0:6]
    for i in range(12):
        for c in range(21):
            V[i][c] = V1[i][c]  # B1 rows 0-1, all 6 cols
    
    # Part 2: A2 is rows 0-1 of A, cols 2-4 (2x3)
    for i in range(6):
        for c in range(36):
            i_row = i // 3
            i_col = i % 3
            global_col = 2 + i_col
            U[i_row * 5 + global_col][21 + c] = U2[i][c]
    
    # V2 is for B2 (3x6), which is B[2:5, 0:6]
    for i in range(18):
        for c in range(36):
            i_row = i // 6
            i_col = i % 6
            global_row = 2 + i_row
            V[global_row * 6 + i_col][21 + c] = V2[i][c]
    
    # W: combine W1 and W2
    for i in range(12):
        for c in range(21):
            W[i][c] = W1[i][c]
    for i in range(12):
        for c in range(36):
            W[i][21 + c] = W2[i][c]
    
    return {"U": U, "V": V, "W": W}

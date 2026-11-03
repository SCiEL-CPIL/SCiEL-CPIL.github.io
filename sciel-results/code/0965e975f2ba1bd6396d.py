"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    # We will build the decomposition for 4x4 by 4x7 as a block matrix.
    # Split A (4x4) into 2x2 blocks of size 2x2.
    # Split B (4x7) into 2x2 blocks of size 2x4 and 2x3.
    # Specifically, let B = [B11 B12 B13] where B11, B12 are 4x2 and B13 is 4x3.
    # Then C = A B = [A B11, A B12, A B13].
    
    # We need decompositions for:
    # 1. 4x4 * 4x2 -> rank 28 (derived from 4x 2x2x2 Strassen)
    # 2. 4x4 * 4x3 -> rank 36 (derived from 4x 2x2x3, where 2x2x3 is rank 9)
    
    # Let's define the Strassen decomposition for 2x2 * 2x2 * 2 (rank 7).
    # Input U size 4x7, V size 4x7, W size 4x7.
    # Indices for U: i*m+j -> i in {0,1}, j in {0,1}
    # Indices for V: j*p+k -> j in {0,1}, k in {0,1}
    # Indices for W: k*n+i -> k in {0,1}, i in {0,1}
    
    def get_strassen_222():
        # Returns U, V, W for 2x2 * 2x2 * 2
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11 + A22) * (B11 + B22)
        # A11=U[0,0], A22=U[1,1]
        # B11=V[0,0], B22=V[1,1]
        U[0][0] = 1; U[1][0] = 1
        V[0][0] = 1; V[1][0] = 1
        # C11 += M1, C22 += M1
        # C11 = W[0*2+0] = W[0]
        # C22 = W[1*2+1] = W[3]
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21 + A22) * B11
        # A21=U[1,0], A22=U[1,1]
        # B11=V[0,0]
        U[1][1] = 1; U[1][2] = 1 # Wait, M2 uses A21+A22.
        # Let's be precise.
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B10-B11)
        # M4: A11(B01-B00)
        # M5: (A00+A10)B11
        # M6: (A11-A10)(B00+B10)
        # M7: (A01-A11)(B01+B11)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1
        
        # M2
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[2][1]=1; W[3][1]=1
        
        # M3
        U[0][2]=1
        V[2][2]=1; V[3][2]=-1
        W[0][2]=1; W[2][2]=-1
        
        # M4
        U[3][3]=1
        V[1][3]=1; V[0][3]=-1
        W[1][3]=1; W[3][3]=-1
        
        # M5
        U[0][4]=1; U[2][4]=1
        V[3][4]=1
        W[1][4]=1; W[3][4]=1
        
        # M6
        U[3][5]=1; U[2][5]=-1
        V[0][5]=1; V[2][5]=1
        W[2][5]=1; W[1][5]=-1
        
        # M7
        U[1][6]=1; U[3][6]=-1
        V[1][6]=1; V[3][6]=1
        W[1][6]=1; W[0][6]=-1
        
        return U, V, W
    
    # Now we need 4x4 * 4x2.
    # A is 4x4, B is 4x2.
    # Split A into A_top (2x4) and A_bot (2x4).
    # Split B into B_top (2x2) and B_bot (2x2).
    # C_top = A_top * B = [A_top,0] * [B_top; B_bot] = A_top B_top. (2x2 result)
    # C_bot = A_bot * B = [A_bot,0] * [B_top; B_bot] = A_bot B_bot. (2x2 result)
    # Wait, A is 4x4. B is 4x2.
    # A = [[A11, A12], [A21, A22]] where each block is 2x2.
    # B = [[B11], [B21]] where B11, B21 are 2x2.
    # C = A B. C is 4x2.
    # C_top (2x2) = A11 B11 + A12 B21.
    # C_bot (2x2) = A21 B11 + A22 B21.
    # This requires 4 products of 2x2 * 2x2.
    # Each product has rank 7. Total rank 28.
    
    def get_44x42():
        # Returns U (16x28), V (8x28), W (8x28)
        # U indices: i*m+j, i in 0..3, j in 0..3.
        # V indices: j*p+k, j in 0..3, k in 0..1.
        # W indices: k*n+i, k in 0..1, i in 0..3.
        
        U = [[0]*28 for _ in range(16)]
        V = [[0]*28 for _ in range(8)]
        W = [[0]*28 for _ in range(8)]
        
        U222, V222, W222 = get_strassen_222()
        
        # Product 1: A11 * B11 -> contributes to C_top
        # A11 is A[0:2, 0:2]. Indices in U: 0,1,2,3.
        # B11 is B[0:2, 0:2]. Indices in V: 0,1,4,5.
        # C_top is C[0:2, 0:2]. Indices in W: 0,1, 2,3 (since k=0, i=0,1).
        # W index for 2x2x2 is k*2+i. Here k is 0 or 1 (for C_top, k=0 in the 4x2 context? No).
        # In 4x4x2, W is 2x4. Index k*4+i? No, W[k*n+i]. n=4? No, n in 4x4x2 is 4.
        # Wait, W shape is p*n. p=2, n=4. So W is 8x28.
        # W index: k*4 + i.
        # C_top corresponds to k=0 (first 2 rows of C? No, C is 4x2. Rows 0,1 are top).
        # So k=0 for C_top, k=1 for C_bot?
        # C = [C_top; C_bot].
        # C_top is 2x2. C_bot is 2x2.
        # In W (2x4), column k=0 is C[:,0], column k=1 is C[:,1].
        # So C_top[:,0] is W[0*4 + 0:2, 0] = W[0:2, 0].
        # C_top[:,1] is W[0*4 + 2:4, 0] = W[2:4, 0].
        # Wait, W index is k*n + i.
        # For C_top, i is 0,1 (rows of A top). k is 0,1 (cols of B).
        # So W indices for C_top are k*4 + i.
        # k=0, i=0 -> 0. k=0, i=1 -> 1.
        # k=1, i=0 -> 4. k=1, i=1 -> 5.
        
        # Let's map the 2x2x2 Strassen to these.
        # Strassen U222 indices: 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1].
        # Here A11 is A[0:2, 0:2].
        # So U222[0] -> U[0]
        # U222[1] -> U[1]
        # U222[2] -> U[2]
        # U222[3] -> U[3]
        
        # V222 indices: 0,1,2,3 correspond to B[0,0], B[0,1], B[1,0], B[1,1].
        # Here B11 is B[0:2, 0:2].
        # B index in 4x4x2: j*2+k.
        # B[0,0] -> V[0]
        # B[0,1] -> V[1]
        # B[1,0] -> V[2]
        # B[1,1] -> V[3]
        
        # W222 indices: 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1].
        # Here C_top is C[0:2, 0:2].
        # C index in 4x4x2: k*4+i.
        # C[0,0] (k=0, i=0) -> W[0]
        # C[0,1] (k=1, i=0) -> W[4]
        # C[1,0] (k=0, i=1) -> W[1]
        # C[1,1] (k=1, i=1) -> W[5]
        
        # Mapping:
        # U222 row r -> U row r
        # V222 row c -> V row c
        # W222 row s -> W row map(s)
        # map(0)=0, map(1)=4, map(2)=1, map(3)=5.
        
        for col in range(7):
            for r in range(4):
                if U222[r][col] != 0:
                    U[r][col] += U222[r][col]
            for c in range(4):
                if V222[c][col] != 0:
                    V[c][col] += V222[c][col]
            for s in range(4):
                if W222[s][col] != 0:
                    w_idx = [0, 4, 1, 5][s]
                    W[w_idx][col] += W222[s][col]
        
        # Product 2: A12 * B21
        # A12 is A[0:2, 2:4]. U indices: 4,5,6,7.
        # B21 is B[2:4, 0:2]. V indices: 4,5,6,7.
        # Contributes to C_top.
        # U222 row r -> U row 4+r
        # V222 row c -> V row 4+c
        # W222 row s -> W row map(s)
        
        for col in range(7):
            for r in range(4):
                if U222[r][col] != 0:
                    U[4+r][col+7] += U222[r][col]
            for c in range(4):
                if V222[c][col] != 0:
                    V[4+c][col+7] += V222[c][col]
            for s in range(4):
                if W222[s][col] != 0:
                    w_idx = [0, 4, 1, 5][s]
                    W[w_idx][col+7] += W222[s][col]
        
        # Product 3: A21 * B11
        # A21 is A[2:4, 0:2]. U indices: 8,9,10,11.
        # B11 is B[0:2, 0:2]. V indices: 0,1,2,3.
        # Contributes to C_bot.
        # C_bot is C[2:4, 0:2].
        # C index in 4x4x2: k*4+i.
        # C[2,0] (k=0, i=2) -> W[2]
        # C[2,1] (k=1, i=2) -> W[6]
        # C[3,0] (k=0, i=3) -> W[3]
        # C[3,1] (k=1, i=3) -> W[7]
        # map_bot(0)=2, map_bot(1)=6, map_bot(2)=3, map_bot(3)=7.
        
        for col in range(7):
            for r in range(4):
                if U222[r][col] != 0:
                    U[8+r][col+14] += U222[r][col]
            for c in range(4):
                if V222[c][col] != 0:
                    V[c][col+14] += V222[c][col]
            for s in range(4):
                if W222[s][col] != 0:
                    w_idx = [2, 6, 3, 7][s]
                    W[w_idx][col+14] += W222[s][col]
        
        # Product 4: A22 * B21
        # A22 is A[2:4, 2:4]. U indices: 12,13,14,15.
        # B21 is B[2:4, 0:2]. V indices: 4,5,6,7.
        # Contributes to C_bot.
        
        for col in range(7):
            for r in range(4):
                if U222[r][col] != 0:
                    U[12+r][col+21] += U222[r][col]
            for c in range(4):
                if V222[c][col] != 0:
                    V[4+c][col+21] += V222[c][col]
            for s in range(4):
                if W222[s][col] != 0:
                    w_idx = [2, 6, 3, 7][s]
                    W[w_idx][col+21] += W222[s][col]
        
        return U, V, W
    
    # Now 4x4 * 4x3.
    # A is 4x4. B is 4x3.
    # Split B into B1 (4x2) and B2 (4x1).
    # C = [A B1, A B2].
    # A B1 is 4x2. Rank 28.
    # A B2 is 4x1. Rank 8 (4x 2x2x1? No. 4x4x1. Split into 2x2x1 products. 4 * 2 = 8).
    # Total rank 36.
    
    def get_44x41():
        # Returns U (16x8), V (4x8), W (4x8)
        # W index k*4+i. k=0. So W is 4x8.
        U = [[0]*8 for _ in range(16)]
        V = [[0]*8 for _ in range(4)]
        W = [[0]*8 for _ in range(4)]
        
        # 2x2 * 2x1 * 2. Rank 2.
        # A11 * B11 -> C11
        # A21 * B11 -> C21
        # Wait, B is 2x1.
        # C1 = A11 B1 + A12 B2.
        # C2 = A21 B1 + A22 B2.
        # Products:
        # 1. A11 * B1 (2x2 * 2x1)
        # 2. A12 * B2 (2x2 * 2x1)
        # 3. A21 * B1
        # 4. A22 * B2
        
        # 2x2 * 2x1 decomposition.
        # U221 (4x2), V221 (2x2), W221 (2x2).
        # U221[0,0]=1, U221[0,1]=0, ...
        # Actually, 2x2 * 2x1 is just 2 independent 2x1 * 2x1? No.
        # C1 = A11 B1 + A12 B2.
        # We can just do schoolbook for the 2x2 blocks?
        # A11 * B1 requires 4 mults? No, 2x2 * 2x1.
        # (a b; c d) * (e; f) = (ae+bf; ce+df).
        # Can be done in 2 mults?
        # m1 = a*f, m2 = (a+b)*e? No.
        # m1 = a*f, m2 = b*e. C1 = m1+m2.
        # m3 = c*f, m4 = d*e. C2 = m3+m4.
        # Total 4 mults.
        # Is there a rank 2 decomposition?
        # (a e, b f, c e, d f).
        # Tensor is sum of 4 rank 1.
        # Can we do it in 2?
        # If rank 2, then U, V, W are 4x2, 2x2, 2x2.
        # C1 = u10 v10 w10 + u11 v11 w11.
        # It turns out rank of 2x2x1 is 2.
        # Proof: C1 = (a+b)(e+f) - b f - a e? No.
        # Let's use rank 4 for safety, or find rank 2.
        # Actually, 2x2 * 2x1 is bilinear.
        # a e, b f, c e, d f.
        # (a+b)e, (c+d)f? No.
        # Let's assume rank 4 for 2x2x1.
        # Then 4 * 4 = 16 for 4x4x1.
        # Previous estimate 8 was based on rank 2.
        # If rank is 4, then 4x4x1 is 16.
        # Then 4x4x3 is 28 + 16 = 44.
        # Total 28+28+44 = 100.
        # Still better than 112.
        
        # Let's verify rank of 2x2x1.
        # Matrix [[a, b], [c, d]] * [[e], [f]] = [[ae+bf], [ce+df]].
        # Tensor T = a e u1 v1 w1 + b f u2 v2 w2 + c e u3 v3 w3 + d f u4 v4 w4.
        # Can we combine?
        # (a+b) e, (c+d) f?
        # m1 = (a+b) e = ae + be.
        # m2 = (c+d) f = cf + df.
        # m3 = b f.
        # m4 = a e?
        # C1 = ae + bf = m1 - be + bf. No.
        
        # Actually, rank of M_{2,2,1} is 2.
        # U = [[1,0],[0,0],[0,0],[1,0]]?
        # Let's look it up or derive.
        # C1 = a e + b f.
        # C2 = c e + d f.
        # Let m1 = e(a+b), m2 = f(c+d).
        # C1 = m1 - b e + b f.
        # This doesn't seem to reduce.
        
        # However, we can use the fact that we only need to add to the existing decomposition.
        # For 4x4x1, we can just use schoolbook 16 mults.
        
        U = [[0]*16 for _ in range(16)]
        V = [[0]*16 for _ in range(4)]
        W = [[0]*16 for _ in range(4)]
        
        for i in range(4):
            for j in range(4):
                col = i*4+j
                U[i*4+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1 # W index k*4+i, k=0. So W[i].
        
        return U, V, W

    # Construct final 4x4x7
    # Rank = 28 (for 4x2) + 28 (for 4x2) + 16 (for 4x1) = 72.
    # Wait, B is 4x7.
    # Split B into B1 (4x2), B2 (4x2), B3 (4x3).
    # B3 = B3a (4x2) + B3b (4x1).
    # So 3 * 4x2 and 1 * 4x1.
    # Rank = 3 * 28 + 16 = 84 + 16 = 100.
    
    # Let's check 4x4x2 rank again.
    # 4 * 7 = 28.
    # 3 * 28 = 84.
    # 4x4x1 rank 16.
    # Total 100.
    
    # Can we do better for 4x4x1?
    # If rank 2x2x1 is 2, then 4*2=8.
    # Total 3*28 + 8 = 92.
    
    # Let's try to implement 2x2x1 rank 2.
    # C1 = ae + bf
    # C2 = ce + df
    # m1 = e(a+b)
    # m2 = f(c+d)
    # m3 = bf
    # m4 = ae
    # C1 = m1 - be + m3? No.
    
    # Actually, rank of 2x2x1 is 2.
    # U = [[1, 1], [0, 0], [0, 0], [1, 1]]?
    # Let's assume rank 4 for safety, or just use schoolbook.
    # Schoolbook is 16.
    
    # Let's just use the 100 rank decomposition.
    
    U_442, V_442, W_442 = get_44x42()
    U_441, V_441, W_441 = get_44x41()
    
    rank = 3 * 28 + 16
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)] # 4*7
    W = [[0]*rank for _ in range(28)] # 7*4
    
    # B is 4x7.
    # B1 cols 0,1. V indices 0,1,4,5.
    # B2 cols 2,3. V indices 2,3,6,7.
    # B3a cols 4,5. V indices 8,9,12,13.
    # B3b col 6. V indices 10,14.
    
    # 1. B1 (4x2)
    for col in range(28):
        for r in range(16):
            U[r][col] = U_442[r][col]
        for c in range(8):
            # V_442 is 8x28.
            # V_442[c] corresponds to B[0:2, 0:2] and B[2:4, 0:2].
            # B1 is B[0:2, 0:2] and B[2:4, 0:2].
            # So V_442[c] maps directly to V[c].
            V[c][col] = V_442[c][col]
        for k in range(2):
            for i in range(4):
                W[k*4+i][col] = W_442[k*4+i][col]
    
    # 2. B2 (4x2)
    for col in range(28):
        for r in range(16):
            U[r][col+28] = U_442[r][col]
        for c in range(8):
            # V_442[c] maps to V[c+8] ?
            # B2 is B[0:2, 2:4] and B[2:4, 2:4].
            # V index for B2: j*4+k.
            # j=0, k=2,3 -> 2,3.
            # j=1, k=2,3 -> 6,7.
            # V_442 uses 0,1,4,5.
            # So V_442[0] -> V[2], V_442[1] -> V[3], V_442[4] -> V[6], V_442[5] -> V[7].
            # V_442[2] -> V[2+4]? No.
            # V_442 row c corresponds to B[j, k] where c = j*2+k.
            # For B2, j is 0,1 but k is 2,3.
            # So V_442[0] (j=0,k=0) -> V[0*4+2] = V[2].
            # V_442[1] (j=0,k=1) -> V[0*4+3] = V[3].
            # V_442[2] (j=1,k=0) -> V[1*4+2] = V[6].
            # V_442[3] (j=1,k=1) -> V[1*4+3] = V[7].
            # V_442[4] (j=0,k=0 of second block?) No, V_442 is for 4x2.
            # V_442 indices:
            # 0,1 for B11 (j=0, k=0,1)
            # 4,5 for B21 (j=1, k=0,1)
            # Wait, my get_44x42 V was 8x28.
            # V_442[0] = B[0,0], V_442[1] = B[0,1], V_442[2] = B[1,0], V_442[3] = B[1,1]?
            # No, V_442 was built for 4x2 B.
            # V index j*2+k.
            # j=0,1. k=0,1.
            # So V_442[0]=B[0,0], V_442[1]=B[0,1], V_442[2]=B[1,0], V_442[3]=B[1,1].
            # V_442[4]=B[2,0], V_442[5]=B[2,1], V_442[6]=B[3,0], V_442[7]=B[3,1].
            
            # For B2 (cols 2,3):
            # B[0,2] -> V[2]. V_442[0] -> V[2].
            # B[0,3] -> V[3]. V_442[1] -> V[3].
            # B[1,2] -> V[6]. V_442[2] -> V[6].
            # B[1,3] -> V[7]. V_442[3] -> V[7].
            # B[2,2] -> V[10]. V_442[4] -> V[10].
            # B[2,3] -> V[11]. V_442[5] -> V[11].
            # B[3,2] -> V[14]. V_442[6] -> V[14].
            # B[3,3] -> V[15]. V_442[7] -> V[15].
            
            map_V = [2, 3, 6, 7, 10, 11, 14, 15]
            V[map_V[c]][col+28] = V_442[c][col]
            
        for k in range(2):
            for i in range(4):
                # W index k*4+i.
                # Here k is 2,3 for B2? No, W is 7x4.
                # W index is k_out * 4 + i.
                # k_out is 0..6.
                # For B2, k_out is 2,3.
                # So W[2*4+i] and W[3*4+i].
                # W_442 has k=0,1.
                # W_442[0*4+i] -> W[2*4+i].
                # W_442[1*4+i] -> W[3*4+i].
                W[2*4+i][col+28] = W_442[0*4+i][col]
                W[3*4+i][col+28] = W_442[1*4+i][col]
    
    # 3. B3a (4x2, cols 4,5)
    for col in range(28):
        for r in range(16):
            U[r][col+56] = U_442[r][col]
        map_V = [4, 5, 8, 9, 12, 13, 16, 17]
        for c in range(8):
            V[map_V[c]][col+56] = V_442[c][col]
        for k in range(2):
            for i in range(4):
                W[4*4+i][col+56] = W_442[0*4+i][col]
                W[5*4+i][col+56] = W_442[1*4+i][col]
    
    # 4. B3b (4x1, col 6)
    for col in range(16):
        for r in range(16):
            U[r][col+84] = U_441[r][col]
        for c in range(4):
            # V_441 is 4x16. V_441[c] corresponds to B[c,0].
            # B3b is B[:, 6].
            # So V_441[0] -> V[6], V_441[1] -> V[10], V_441[2] -> V[14], V_441[3] -> V[18].
            map_V = [6, 10, 14, 18]
            V[map_V[c]][col+84] = V_441[c][col]
        for i in range(4):
            W[6*4+i][col+84] = W_441[i][col]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

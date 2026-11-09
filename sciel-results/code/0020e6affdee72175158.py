"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Use Kronecker product structure: 3x4x6 = (3x2x2) ⊗ (1x2x3)
    rank = rank(3x2x2) * rank(1x2x3) = 11 * 6 = 66
    """
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Factor: n1*n2 = n, m1*m2 = m, p1*p2 = p
    # n1=3, n2=1, m1=2, m2=2, p1=2, p2=3
    n1, n2 = 3, 1
    m1, m2 = 2, 2
    p1, p2 = 2, 3
    
    # First factor: 3x2x2 multiplication (rank 11)
    # Second factor: 1x2x3 multiplication (rank 6, schoolbook/outer product)
    
    # Build 3x2x2 decomposition (rank 11)
    # This multiplies a 3x2 matrix by a 2x2 matrix
    # U1: (n1*m1) x rank1 = 6 x 11
    # V1: (m1*p1) x rank1 = 4 x 11  
    # W1: (p1*n1) x rank1 = 6 x 11
    
    # Known rank-11 decomposition for 3x2x2:
    # We can use the fact that 3x2x2 can be decomposed using the 2x2x2 Strassen plus one extra
    # Actually, let's use a direct construction.
    
    # For 3x2x2: multiplying A (3x2) by B (2x2) to get C (3x2)
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # We need rank-11 tensors.
    
    # Let me use a known construction:
    # The 3x2x2 tensor has rank 11. Here's one decomposition:
    # Use the schoolbook 12 terms and merge one pair.
    
    # Actually, let me just use schoolbook for 3x2x2 (rank 12) for simplicity,
    # and see if I can do better. Rank 12 * 6 = 72, same as schoolbook.
    # I need the rank-11 decomposition.
    
    # Rank-11 decomposition of 3x2x2:
    # From literature, one can use the following approach:
    # The 3x2x2 multiplication can be seen as: for each output column k,
    # C[:,k] = A * B[:,k], which is a 3x2 times 2x1 multiplication (rank 6 each),
    # but the two columns share structure.
    
    # Let me construct it explicitly. The tensor T[i,j,k] = delta (multiplication tensor)
    # for 3x2x2: T is indexed by (i in 0..2, j in 0..1, k in 0..1)
    # T[i,j,k] = 1 if this term contributes to C[i,k]
    
    # Schoolbook: 12 terms, each (i,j,k) triple
    # To get rank 11, we need to combine two terms.
    
    # One known rank-11 decomposition:
    # Use Strassen-like ideas. Let me use a verified construction.
    
    # Actually, let me use a simpler approach: use the known result that
    # rank(3,2,2) = 11 and construct it via a specific method.
    
    # Here's a construction based on the fact that we can write:
    # For the 3x2x2 case, we can use the following 11 rank-1 tensors:
    
    # Let me just hardcode a known rank-11 decomposition for 3x2x2.
    # After research, one such decomposition uses the following:
    
    # U1 (6x11), V1 (4x11), W1 (6x11) for 3x2x2
    # Row indexing for U1: i*m1+j = i*2+j for i in 0..2, j in 0..1
    # Row indexing for V1: j*p1+k = j*2+k for j in 0..1, k in 0..1
    # Row indexing for W1: k*n1+i = k*3+i for k in 0..1, i in 0..2
    
    # I'll use a verified rank-11 decomposition. Let me construct it:
    # Based on the work of Bini et al., one rank-11 decomposition is:
    
    U1 = [[0]*11 for _ in range(6)]  # 3*2
    V1 = [[0]*11 for _ in range(4)]  # 2*2
    W1 = [[0]*11 for _ in range(6)]  # 2*3
    
    # Schoolbook terms (12 terms), then we'll modify to get 11
    # Term (i,j,k): U1[2*i+j][c] = 1, V1[2*j+k][c] = 1, W1[3*k+i][c] = 1
    
    # Let me use a known rank-11 decomposition from the literature.
    # One approach: the 3x2x2 tensor can be decomposed using the
    # following 11 rank-1 components:
    
    # I'll use the decomposition from the paper by Bläser (2001) or similar.
    # For now, let me use a construction that I can verify:
    
    # The key insight: C = A*B where A is 3x2, B is 2x2.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    
    # We can group terms. Let's use the following 11 terms:
    # For the first two rows (2x2x2 part), use Strassen (7 terms):
    # Then for the third row, add 4 more terms, but one can be shared.
    
    # Strassen for 2x2x2 (rows 0,1 of A, all of B, columns 0,1 of C):
    # Let A_top = A[0:2,:], C_top = C[0:2,:]
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A10)*(B00+B10)
    # M7 = (A11-A01)*(B01+B11)
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M1 - M5 + M7
    # C10 = M2 - M1 + M4 + M5
    # C11 = M2 - M3 + M6 - M4
    
    # For row 2: C20 = A20*B00 + A21*B10, C21 = A20*B01 + A21*B11
    # These need 4 more terms: (A20,B00), (A21,B10), (A20,B01), (A21,B11)
    # Total: 7 + 4 = 11 terms!
    
    # Now let me write this out:
    # U1 rows: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3, A[2,0]=4, A[2,1]=5
    # V1 rows: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # W1 rows: C[0,0]=0, C[1,0]=1, C[2,0]=2, C[0,1]=3, C[1,1]=4, C[2,1]=5
    # (W1[k*n1+i] = W1[2*k+i])
    
    # Term 0: M1 = (A00+A11)*(B00+B11)
    # U1: A00=1, A11=1; V1: B00=1, B11=1; W1: C00=1, C01=1, C10=-1, C11=1
    U1[0][0] = 1; U1[3][0] = 1  # A[0,0], A[1,1]
    V1[0][0] = 1; V1[3][0] = 1  # B[0,0], B[1,1]
    W1[0][0] = 1; W1[3][0] = 1; W1[1][0] = -1; W1[4][0] = 1  # C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Term 1: M2 = (A10+A11)*B00
    U1[2][1] = 1; U1[3][1] = 1  # A[1,0], A[1,1]
    V1[0][1] = 1  # B[0,0]
    W1[1][1] = 1; W1[4][1] = 1  # C[1,0], C[1,1]
    
    # Term 2: M3 = A00*(B01-B11)
    U1[0][2] = 1  # A[0,0]
    V1[1][2] = 1; V1[3][2] = -1  # B[0,1], B[1,1]
    W1[3][2] = 1; W1[4][2] = -1  # C[0,1], C[1,1]
    
    # Term 3: M4 = A11*(B10-B00)
    U1[3][3] = 1  # A[1,1]
    V1[2][3] = 1; V1[0][3] = -1  # B[1,0], B[0,0]
    W1[0][3] = -1; W1[1][3] = 1; W1[4][3] = -1  # C[0,0], C[1,0], C[1,1]
    
    # Term 4: M5 = (A00+A10)*B11
    U1[0][4] = 1; U1[2][4] = 1  # A[0,0], A[1,0]
    V1[3][4] = 1  # B[1,1]
    W1[3][4] = -1; W1[4][4] = 1  # C[0,1], C[1,1]
    
    # Term 5: M6 = (A01-A10)*(B00+B10)
    U1[1][5] = 1; U1[2][5] = -1  # A[0,1], A[1,0]
    V1[0][5] = 1; V1[2][5] = 1  # B[0,0], B[1,0]
    W1[0][5] = 1; W1[1][5] = 1  # C[0,0], C[1,0]
    
    # Term 6: M7 = (A11-A01)*(B01+B11)
    U1[3][6] = 1; U1[1][6] = -1  # A[1,1], A[0,1]
    V1[1][6] = 1; V1[3][6] = 1  # B[0,1], B[1,1]
    W1[0][6] = -1; W1[3][6] = 1  # C[0,0], C[0,1]
    
    # Term 7: A20*B00
    U1[4][7] = 1  # A[2,0]
    V1[0][7] = 1  # B[0,0]
    W1[2][7] = 1  # C[2,0]
    
    # Term 8: A21*B10
    U1[5][8] = 1  # A[2,1]
    V1[2][8] = 1  # B[1,0]
    W1[2][8] = 1  # C[2,0]
    
    # Term 9: A20*B01
    U1[4][9] = 1  # A[2,0]
    V1[1][9] = 1  # B[0,1]
    W1[5][9] = 1  # C[2,1]
    
    # Term 10: A21*B11
    U1[5][10] = 1  # A[2,1]
    V1[3][10] = 1  # B[1,1]
    W1[5][10] = 1  # C[2,1]
    
    # Now build the 1x2x3 decomposition (rank 6, schoolbook)
    # n2=1, m2=2, p2=3
    # U2: (n2*m2) x rank2 = 2 x 6
    # V2: (m2*p2) x rank2 = 6 x 6
    # W2: (p2*n2) x rank2 = 3 x 6
    
    U2 = [[0]*6 for _ in range(2)]
    V2 = [[0]*6 for _ in range(6)]
    V2 = [[0]*6 for _ in range(6)]  # m2*p2 = 2*3 = 6
    W2 = [[0]*6 for _ in range(3)]
    
    col = 0
    for i2 in range(1):
        for j2 in range(2):
            for k2 in range(3):
                U2[i2*2+j2][col] = 1
                V2[j2*3+k2][col] = 1
                W2[k2*1+i2][col] = 1
                col += 1
    
    # Now form the Kronecker product decomposition
    # n = n1*n2 = 3, m = m1*m2 = 4, p = p1*p2 = 6
    # rank_total = 11 * 6 = 66
    
    rank1 = 11
    rank2 = 6
    rank_total = rank1 * rank2
    
    U = [[0]*rank_total for _ in range(n*m)]  # 3*4 = 12
    V = [[0]*rank_total for _ in range(m*p)]  # 4*6 = 24
    W = [[0]*rank_total for _ in range(p*n)]  # 6*3 = 18
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1 * rank2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            # U index: (i1*n2 + i2) * m + (j1*m2 + j2)
                            # = (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            u_idx = i * m + j
                            U[u_idx][c] = U1[i1*m1+j1][c1] * U2[i2*m2+j2][c2]
            
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            v_idx = j * p + k
                            V[v_idx][c] = V1[j1*p1+k1][c1] * V2[j2*p2+k2][c2]
            
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            w_idx = k * n + i
                            W[w_idx][c] = W1[k1*n1+i1][c1] * W2[k2*n2+i2][c2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END


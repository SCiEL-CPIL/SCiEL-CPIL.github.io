import time

def construct():
    # We use a block decomposition: A is 2x5, B is 5x6, C is 2x6.
    # Split B into 3 blocks of 2 columns each: B = [B0, B1, B2], each Bk is 5x2.
    # C = A B = [A B0, A B1, A B2].
    # For each block k, C_k = A B_k is 2x2.
    # We can use Strassen for each 2x2 multiplication, but A is 2x5, not 2x2.
    # Instead, we split A into a left 2x2 part and a right 2x3 part:
    # A = [A_L, A_R] where A_L is 2x2 (columns 0,1) and A_R is 2x3 (columns 2,3,4).
    # Similarly, B_k = [B_{k,L}; B_{k,R}] where B_{k,L} is 2x2 (rows 0,1) and B_{k,R} is 3x2 (rows 2,3,4).
    # Then A B_k = A_L B_{k,L} + A_R B_{k,R}.
    # A_L B_{k,L} is 2x2, done with Strassen (rank 7).
    # A_R B_{k,R} is 2x3 times 3x2 = 2x2, done with schoolbook (rank 6).
    # So each block k uses 7 + 6 = 13 rank-1 terms.
    # Total rank = 3 * 13 = 39.
    
    n, m, p = 2, 5, 6
    rank = 39
    U = [[0]*rank for _ in range(n*m)]  # 10 x 39
    V = [[0]*rank for _ in range(m*p)]  # 30 x 39
    W = [[0]*rank for _ in range(p*n)]  # 12 x 39
    
    col = 0
    for k in range(3):  # 3 blocks of 2 columns each
        # Strassen part for A_L (2x2) * B_{k,L} (2x2)
        # A_L = [[a00, a01], [a10, a11]]
        # B_{k,L} = [[b00, b01], [b10, b11]] where b_ij = B[j, k*2 + i]
        # Strassen products:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11) b00
        # M3 = a00 (b10-b11)
        # M4 = a11 (b00-b10)
        # M5 = (a00+a01) b11
        # M6 = (a10-a00) (b00+b01)
        # M7 = (a01-a11) (b10+b11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1
        U[0][col] = 1; U[3][col] = 1  # a00, a11
        V[k*2 + 0][col] = 1; V[k*2 + 1][col] = 1  # b00, b11 (row 0,1 of block k)
        W[k*2 + 0][col] = 1; W[k*2 + 1][col] = 1  # C00, C11
        col += 1
        
        # M2
        U[3][col] = 1  # a11 (a10+a11 -> a10 is 1, a11 is 1? No: M2=(a10+a11)*b00)
        # Correction: M2 = (a10 + a11) * b00
        U[3][col] = 1  # a11 index: i=1,j=0 -> 1*2+0=3? No: i*m+j. i=1,j=0 -> 1*2+0=3. a10 is i=1,j=0 -> index 3. a11 is i=1,j=1 -> index 4.
        # Let's re-index carefully.
        # A[i,j], i in {0,1}, j in {0..4}. Index = i*5 + j.
        # a00: 0, a01: 1, a02: 2, a03: 3, a04: 4
        # a10: 5, a11: 6, a12: 7, a13: 8, a14: 9
        # So A_L = [[a00, a01], [a10, a11]] = indices 0,1,5,6.
        # B[j, k], j in {0..4}, k in {0..5}. Index = j*6 + k.
        # For block k (k=0,1,2), columns are 2k, 2k+1.
        # B_{k,L} = rows 0,1 of B, columns 2k, 2k+1.
        # b00 = B[0, 2k] -> index 0*6 + 2k = 2k
        # b01 = B[0, 2k+1] -> index 2k+1
        # b10 = B[1, 2k] -> index 6 + 2k
        # b11 = B[1, 2k+1] -> index 6 + 2k + 1
        
        # Let's redo M1: (a00+a11)(b00+b11)
        # a00: 0, a11: 6. b00: 2k, b11: 7+2k.
        U[0][col] = 1; U[6][col] = 1
        V[2*k][col] = 1; V[7+2*k][col] = 1
        W[2*k][col] = 1; W[2*k+1][col] = 1  # C[k*2+0], C[k*2+1] -> indices 2k, 2k+1
        col += 1
        
        # M2: (a10+a11) b00 = (a5+a6) * b(2k)
        U[5][col] = 1; U[6][col] = 1
        V[2*k][col] = 1
        W[2*k+1][col] = 1  # C10 -> index 2k+1
        col += 1
        
        # M3: a00 (b10-b11) = a0 * (b(7+2k) - b(6+2k))
        U[0][col] = 1
        V[7+2*k][col] = 1; V[6+2*k][col] = -1
        W[2*k+1][col] = 1  # C01 -> index 2k+1
        col += 1
        
        # M4: a11 (b00-b10) = a6 * (b(2k) - b(6+2k))
        U[6][col] = 1
        V[2*k][col] = 1; V[6+2*k][col] = -1
        W[2*k][col] = 1; W[2*k+1][col] = 1  # C00, C10
        col += 1
        
        # M5: (a00+a01) b11 = (a0+a1) * b(7+2k)
        U[0][col] = 1; U[1][col] = 1
        V[7+2*k][col] = 1
        W[2*k][col] = -1; W[2*k+1][col] = 1  # -C00, C01
        col += 1
        
        # M6: (a10-a00) (b00+b01) = (a5-a0) * (b(2k)+b(2k+1))
        U[5][col] = 1; U[0][col] = -1
        V[2*k][col] = 1; V[2*k+1][col] = 1
        W[2*k+1][col] = 1  # C11 -> index 2k+1
        col += 1
        
        # M7: (a01-a11) (b10+b11) = (a1-a6) * (b(6+2k)+b(7+2k))
        U[1][col] = 1; U[6][col] = -1
        V[6+2*k][col] = 1; V[7+2*k][col] = 1
        W[2*k][col] = 1  # C00 -> index 2k
        col += 1
        
        # Schoolbook part for A_R (2x3) * B_{k,R} (3x2)
        # A_R = [[a02, a03, a04], [a12, a13, a14]] indices 2,3,4,7,8,9
        # B_{k,R} = rows 2,3,4 of B, columns 2k, 2k+1
        # b20 = B[2, 2k] -> 12+2k, b21 = B[2, 2k+1] -> 13+2k
        # b30 = B[3, 2k] -> 18+2k, b31 = B[3, 2k+1] -> 19+2k
        # b40 = B[4, 2k] -> 24+2k, b41 = B[4, 2k+1] -> 25+2k
        # C_k,R,00 = a02*b20 + a03*b30 + a04*b40
        # C_k,R,01 = a02*b21 + a03*b31 + a04*b41
        # C_k,R,10 = a12*b20 + a13*b30 + a14*b40
        # C_k,R,11 = a12*b21 + a13*b31 + a14*b41
        
        # Term 1: a02 * b20
        U[2][col] = 1
        V[12+2*k][col] = 1
        W[2*k][col] = 1  # C00
        col += 1
        
        # Term 2: a02 * b21
        U[2][col] = 1
        V[13+2*k][col] = 1
        W[2*k+1][col] = 1  # C01
        col += 1
        
        # Term 3: a03 * b30
        U[3][col] = 1
        V[18+2*k][col] = 1
        W[2*k][col] = 1  # C00
        col += 1
        
        # Term 4: a03 * b31
        U[3][col] = 1
        V[19+2*k][col] = 1
        W[2*k+1][col] = 1  # C01
        col += 1
        
        # Term 5: a04 * b40
        U[4][col] = 1
        V[24+2*k][col] = 1
        W[2*k][col] = 1  # C00
        col += 1
        
        # Term 6: a04 * b41
        U[4][col] = 1
        V[25+2*k][col] = 1
        W[2*k+1][col] = 1  # C01
        col += 1
        
        # Term 7: a12 * b20
        U[7][col] = 1
        V[12+2*k][col] = 1
        W[2*k][col] = 1  # C10 -> index 2k
        col += 1
        
        # Term 8: a12 * b21
        U[7][col] = 1
        V[13+2*k][col] = 1
        W[2*k+1][col] = 1  # C11 -> index 2k+1
        col += 1
        
        # Term 9: a13 * b30
        U[8][col] = 1
        V[18+2*k][col] = 1
        W[2*k][col] = 1  # C10
        col += 1
        
        # Term 10: a13 * b31
        U[8][col] = 1
        V[19+2*k][col] = 1
        W[2*k+1][col] = 1  # C11
        col += 1
        
        # Term 11: a14 * b40
        U[9][col] = 1
        V[24+2*k][col] = 1
        W[2*k][col] = 1  # C10
        col += 1
        
        # Term 12: a14 * b41
        U[9][col] = 1
        V[25+2*k][col] = 1
        W[2*k+1][col] = 1  # C11
        col += 1
    
    return {"U": U, "V": V, "W": W}

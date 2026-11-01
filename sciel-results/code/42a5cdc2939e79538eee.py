# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block decomposition:
    # A = [[A22, a], [b, c]]
    # B = [[B22, d], [e, f]]
    # C = [[C22, c12], [c21, c22]]
    
    # Indices:
    # A22: (0,0), (0,1), (1,0), (1,1) -> U rows 0,1,3,4
    # a: (0,2), (1,2) -> U rows 2,5
    # b: (2,0), (2,1) -> U rows 6,7
    # c: (2,2) -> U row 8
    
    # B22: V rows 0,1,3,4
    # d: V rows 2,5
    # e: V rows 6,7
    # f: V row 8
    
    # C22: W rows 0,1,3,4
    # c12: W rows 2,5
    # c21: W rows 6,7
    # c22: W row 8
    
    col = 0
    
    # 1. A22 * B22 via Strassen (7 mults)
    # S1 = B22 + e  (Wait, Strassen is for 2x2. Here B22 is 2x2. The "e" in Strassen formula corresponds to the bottom-left block of B, which is 'e' in our 3x3 split? No.
    # Strassen for X*Y where X,Y are 2x2.
    # X = A22, Y = B22.
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # We need to map these to the global indices.
    # X11=A11, X12=A12, X21=A21, X22=A22.
    # Y11=B11, Y12=B12, Y21=B21, Y22=B22.
    
    # M1: (A11+A22)(B11+B22)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[1][col] = -1; W[3][col] = -1; W[4][col] = 1
    col += 1
    
    # M2: (A21+A22)B11
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # M3: A11(B12-B22)
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M4: A22(B21-B11)
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[3][col] = 1; W[4][col] = -1
    col += 1
    
    # M5: (A11+A12)B22
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[1][col] = 1; W[3][col] = 1; W[4][col] = -1
    col += 1
    
    # M6: (A21-A11)(B11+B12)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # M7: (A12-A22)(B21+B22)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # 2. ae (2 mults)
    # a = [A13, A23], e = [B31, B32]
    # ae = A13*B31 + A23*B32
    # This contributes to C11 (top-left 2x2)?
    # C11 = A22*B22 + ae + bd + cf
    # ae is a scalar added to C11? No, C11 is 2x2.
    # Wait, my block definition:
    # A = [[A22, a], [b, c]]
    # a is 2x1. b is 1x2.
    # ae is 2x1 * 1x2 = 2x2 matrix.
    # So ae contributes to all 4 elements of C11.
    # ae = [a1, a2] * [e1, e2] = [[a1e1, a1e2], [a2e1, a2e2]]
    
    # Term 8: a1*e1
    U[2][col] = 1 # A13
    V[6][col] = 1 # B31
    W[0][col] = 1 # C11
    col += 1
    
    # Term 9: a1*e2
    U[2][col] = 1
    V[7][col] = 1 # B32
    W[1][col] = 1 # C12
    col += 1
    
    # Term 10: a2*e1
    U[5][col] = 1 # A23
    V[6][col] = 1
    W[3][col] = 1 # C21
    col += 1
    
    # Term 11: a2*e2
    U[5][col] = 1
    V[7][col] = 1
    W[4][col] = 1 # C22
    col += 1
    
    # 3. bd (2 mults)
    # b = [B31? No, b is row 2 of A? No.
    # A = [[A22, a], [b, c]]
    # b is 1x2. b = [A31, A32].
    # d is 2x1. d = [B13, B23].
    # bd = [b1, b2] * [d1, d2] = [[b1d1, b1d2], [b2d1, b2d2]]
    
    # Term 12: b1*d1 (A31*B13)
    U[6][col] = 1
    V[2][col] = 1
    W[0][col] = 1
    col += 1
    
    # Term 13: b1*d2 (A31*B23)
    U[6][col] = 1
    V[5][col] = 1
    W[1][col] = 1
    col += 1
    
    # Term 14: b2*d1 (A32*B13)
    U[7][col] = 1
    V[2][col] = 1
    W[3][col] = 1
    col += 1
    
    # Term 15: b2*d2 (A32*B23)
    U[7][col] = 1
    V[5][col] = 1
    W[4][col] = 1
    col += 1
    
    # 4. cf (1 mult)
    # c = A33, f = B33
    # cf is scalar. Added to C11?
    # C11 = A22*B22 + ae + bd + cf
    # cf is a scalar, so it adds to all 4 elements of C11?
    # No, C11 is the top-left 2x2 block.
    # The formula for C11 (top-left 2x2) is:
    # C11 = A22*B22 + a*e + b*d + c*f
    # Wait, a is 2x1, e is 1x2. ae is 2x2.
    # b is 1x2, d is 2x1. bd is 2x2.
    # c is 1x1, f is 1x1. cf is 1x1.
    # So cf adds to all 4 elements of C11.
    
    # Term 16: c*f
    U[8][col] = 1
    V[8][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # 5. A22*d (4 mults)
    # A22 is 2x2. d is 2x1.
    # Result is 2x1. Adds to C12 (top-right 2x1).
    # C12 = A22*d + a*f
    
    # Term 17: A11*B13
    U[0][col] = 1
    V[2][col] = 1
    W[2][col] = 1 # C13? No, C12 is col 2 and 5?
    # C12 is the top-right 2x1 block.
    # Indices in C:
    # C11 (0,0), C12 (0,1), C13 (0,2)
    # C21 (1,0), C22 (1,1), C23 (1,2)
    # C31 (2,0), C32 (2,1), C33 (2,2)
    # My W mapping:
    # W row: k*n + i
    # C11 (i=0, k=0) -> W[0]
    # C12 (i=0, k=1) -> W[3]
    # C13 (i=0, k=2) -> W[6]
    # C21 (i=1, k=0) -> W[1]
    # C22 (i=1, k=1) -> W[4]
    # C23 (i=1, k=2) -> W[7]
    # C31 (i=2, k=0) -> W[2]
    # C32 (i=2, k=1) -> W[5]
    # C33 (i=2, k=2) -> W[8]
    
    # Correction:
    # C12 is (i=0, k=1) -> W[3].
    # C22 is (i=1, k=1) -> W[4].
    # So C12 block is W[3] and W[4].
    
    # Term 17: A11*B13 (contributes to C13? No, A22*d)
    # A22 = [[A11, A12], [A21, A22]]
    # d = [B13, B23]
    # A22*d = [A11*B13 + A12*B23, A21*B13 + A22*B23]
    # This is a 2x1 vector.
    # It adds to C12 (which is 2x1).
    # C12 = [C13, C23]? No.
    # C = [[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]]
    # My block C12 was defined as the top-right 2x1 of the 3x3?
    # No, my block split was:
    # A = [[A22, a], [b, c]]
    # A22 is 2x2 (top-left).
    # a is 2x1 (top-right).
    # b is 1x2 (bottom-left).
    # c is 1x1 (bottom-right).
    
    # So C12 (top-right 2x1) corresponds to C13 and C23 in the 3x3 matrix.
    # C13 is (i=0, k=2) -> W[6].
    # C23 is (i=1, k=2) -> W[7].
    
    # Term 17: A11*B13
    U[0][col] = 1
    V[2][col] = 1
    W[6][col] = 1
    col += 1
    
    # Term 18: A12*B23
    U[1][col] = 1
    V[5][col] = 1
    W[6][col] = 1
    col += 1
    
    # Term 19: A21*B13
    U[3][col] = 1
    V[2][col] = 1
    W[7][col] = 1
    col += 1
    
    # Term 20: A22*B23
    U[4][col] = 1
    V[5][col] = 1
    W[7][col] = 1
    col += 1
    
    # 6. a*f (2 mults)
    # a = [A13, A23], f = B33
    # a*f = [A13*B33, A23*B33]
    # Adds to C12 (C13, C23).
    
    # Term 21: A13*B33
    U[2][col] = 1
    V[8][col] = 1
    W[6][col] = 1
    col += 1
    
    # Term 22: A23*B33
    U[5][col] = 1
    V[8][col] = 1
    W[7][col] = 1
    col += 1
    
    # 7. b*B22 (4 mults)
    # b = [A31, A32], B22 = [[B11, B12], [B21, B22]]
    # b*B22 = [A31*B11 + A32*B21, A31*B12 + A32*B22]
    # Adds to C21 (bottom-left 1x2).
    # C21 corresponds to C31 and C32.
    # C31 is (i=2, k=0) -> W[2].
    # C32 is (i=2, k=1) -> W[5].
    
    # Term 23: A31*B11
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # Term 24: A32*B21
    U[7][col] = 1
    V[3][col] = 1
    W[2][col] = 1
    col += 1
    
    # Term 25: A31*B12
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # Term 26: A32*B22
    U[7][col] = 1
    V[4][col] = 1
    W[5][col] = 1
    col += 1
    
    # 8. c*e (2 mults)
    # c = A33, e = [B31, B32]
    # c*e = [A33*B31, A33*B32]
    # Adds to C21 (C31, C32).
    
    # Term 27: A33*B31
    U[8][col] = 1
    V[6][col] = 1
    W[2][col] = 1
    col += 1
    
    # Term 28: A33*B32
    U[8][col] = 1
    V[7][col] = 1
    W[5][col] = 1
    col += 1
    
    # 9. b*f (2 mults)
    # b = [A31, A32], f = B33
    # b*f = [A31*B33, A32*B33]
    # Adds to C22 (bottom-right 1x1).
    # C22 is (i=2, k=2) -> W[8].
    
    # Term 29: A31*B33
    U[6][col] = 1
    V[8][col] = 1
    W[8][col] = 1
    col += 1
    
    # Term 30: A32*B33
    U[7][col] = 1
    V[8][col] = 1
    W[8][col] = 1
    col += 1
    
    # Wait, I counted 30 terms.
    # 7 (Strassen) + 4 (ae) + 4 (bd) + 1 (cf) + 4 (A22d) + 2 (af) + 4 (bB22) + 2 (ce) + 2 (bf) = 30.
    # 30 is worse than 27.
    
    # I made a mistake in the count or the decomposition.
    # The naive block approach is not better.
    
    # I will return the seed.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

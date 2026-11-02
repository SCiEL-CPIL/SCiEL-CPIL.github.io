import numpy as np

def construct():
    n, m, p = 2, 3, 3
    
    # We will construct a rank 17 decomposition.
    # Split A into A1 (2x2) and A2 (2x1)
    # Split B into B1 (2x3) and B2 (1x3)
    # C = A1*B1 + A2*B2
    
    # Part 1: A2*B2 (2x1 * 1x3) -> Rank 6
    # Part 2: A1*B1 (2x2 * 2x3) -> Rank 11 (Strassen for 2x2x2 + Schoolbook for 2x2x1)
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)] # 6x17
    V = [[0] * rank for _ in range(m * p)] # 9x17
    W = [[0] * rank for _ in range(p * n)] # 6x17
    
    col = 0
    
    # --- Part 2: A1*B1 ---
    # A1 is rows 0,1 of A, cols 0,1. Indices in A: 0,1,2,3 (i*m+j)
    # A1[0,0]=A[0], A1[0,1]=A[1], A1[1,0]=A[2], A1[1,1]=A[3]
    # B1 is rows 0,1 of B, all cols. Indices in B: 0,1,2, 3,4,5 (j*p+k)
    # B1[0,0]=B[0], B1[0,1]=B[1], B1[0,2]=B[2]
    # B1[1,0]=B[3], B1[1,1]=B[4], B1[1,2]=B[5]
    # C1 is rows 0,1 of C, cols 0,1,2. Indices in C: k*n+i
    # C1[0,0]=C[0], C1[0,1]=C[1], C1[0,2]=C[2]
    # C1[1,0]=C[3], C1[1,1]=C[4], C1[1,2]=C[5]
    
    # Strassen for C1[:, 0:2] (2x2x2)
    # m1 = (a00+a11)(b00+b11)
    # U: a00, a11 -> 1, 1
    # V: b00, b11 -> 1, 1
    # W: c00 += m1, c11 += m1, c01 -= m1, c10 -= m1
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[3][col] = 1; W[1][col] = -1; W[4][col] = -1
    col += 1
    
    # m2 = (a10+a11)b00
    # U: a10, a11 -> 1, 1
    # V: b00 -> 1
    # W: c10 += m2, c11 += m2
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # m3 = a00(b01-b11)
    # U: a00 -> 1
    # V: b01, b11 -> 1, -1
    # W: c01 += m3
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[1][col] = 1
    col += 1
    
    # m4 = a11(b10-b00)
    # U: a11 -> 1
    # V: b10, b00 -> 1, -1
    # W: c10 += m4
    U[3][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[3][col] = 1
    col += 1
    
    # m5 = (a00+a01)b11
    # U: a00, a01 -> 1, 1
    # V: b11 -> 1
    # W: c00 += m5, c01 += m5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # m6 = (a10-a00)(b00+b01)
    # U: a10, a00 -> 1, -1
    # V: b00, b01 -> 1, 1
    # W: c11 += m6
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # m7 = (a01-a11)(b10+b11)
    # U: a01, a11 -> 1, -1
    # V: b10, b11 -> 1, 1
    # W: c10 += m7
    U[1][col] = 1; U[3][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[3][col] = 1
    col += 1
    
    # Schoolbook for C1[:, 2] (2x2x1)
    # c02 = a00*b02 + a01*b12
    # c12 = a10*b02 + a11*b12
    # 4 terms
    # a00*b02
    U[0][col] = 1
    V[2][col] = 1
    W[2][col] = 1
    col += 1
    
    # a01*b12
    U[1][col] = 1
    V[5][col] = 1
    W[2][col] = 1
    col += 1
    
    # a10*b02
    U[2][col] = 1
    V[2][col] = 1
    W[5][col] = 1
    col += 1
    
    # a11*b12
    U[3][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    # --- Part 1: A2*B2 ---
    # A2 is rows 0,1 of A, col 2. Indices: 4, 5
    # B2 is row 2 of B, all cols. Indices: 6, 7, 8
    # C2 is rows 0,1 of C, cols 0,1,2. (Same C indices as before, but added to)
    # c00 += a02*b20
    # c01 += a02*b21
    # c02 += a02*b22
    # c10 += a12*b20
    # c11 += a12*b21
    # c12 += a12*b22
    
    # a02*b20
    U[4][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # a02*b21
    U[4][col] = 1
    V[7][col] = 1
    W[1][col] = 1
    col += 1
    
    # a02*b22
    U[4][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # a12*b20
    U[5][col] = 1
    V[6][col] = 1
    W[3][col] = 1
    col += 1
    
    # a12*b21
    U[5][col] = 1
    V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # a12*b22
    U[5][col] = 1
    V[8][col] = 1
    W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

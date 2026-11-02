"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We express 4x4x6 as (2x2x2) ⊗ (2x2x3)
    # Factor 1: 2x2x2 matrix multiplication, rank 7 (Strassen)
    # Factor 2: 2x2x3 matrix multiplication, rank 8
    
    # Strassen decomposition for 2x2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    # Strassen's 7 products:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U for 2x2x2: shape (2*2, 7) = (4, 7)
    # U[i*2+j, c] where i,j index into A (2x2)
    # V for 2x2x2: shape (2*2, 7) = (4, 7)  
    # V[j*2+k, c] where j,k index into B (2x2)
    # W for 2x2x2: shape (2*2, 7) = (4, 7)
    # W[k*2+i, c] where k,i index into C (2x2), trace convention W[k*n+i]
    
    # For 2x2x2: n=2, m=2, p=2
    # U rows: i*m+j = i*2+j for i,j in 0..1
    # V rows: j*p+k = j*2+k for j,k in 0..1
    # W rows: k*n+i = k*2+i for k,i in 0..1
    
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    # a11 is U[0], a22 is U[3]
    # b11 is V[0], b22 is V[3]
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    # C11 += M1, C22 += M1
    # C11: k=0,i=0 -> W[0*2+0]=W[0]
    # C22: k=1,i=1 -> W[1*2+1]=W[3]
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a21+a22)b11
    # a21 is U[2], a22 is U[3]
    # b11 is V[0]
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    # C21 += M2: k=0,i=1 -> W[0*2+1]=W[1]
    # C22 -= M2: k=1,i=1 -> W[3]
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = a11(b12-b22)
    # a11 is U[0]
    # b12 is V[1], b22 is V[3]
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    # C12 += M3: k=1,i=0 -> W[1*2+0]=W[2]
    # C22 += M3: k=1,i=1 -> W[3]
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = a22(b21-b11)
    # a22 is U[3]
    # b21 is V[2], b11 is V[0]
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    # C11 += M4: W[0]
    # C21 += M4: W[1]
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (a11+a12)b22
    # a11 is U[0], a12 is U[1]
    # b22 is V[3]
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    # C11 -= M5: W[0]
    # C12 += M5: W[2]
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    # a21 is U[2], a11 is U[0]
    # b11 is V[0], b12 is V[1]
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    # C22 += M6: W[3]
    W1[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    # a12 is U[1], a22 is U[3]
    # b21 is V[2], b22 is V[3]
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    # C11 += M7: W[0]
    W1[0][6] = 1
    
    # Now for 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # We'll use a simple decomposition. Rank 8 approach:
    # Treat as two 2x2x2 multiplications for columns 0-1 and a shared part for column 2?
    # Actually, let's use rank 8:
    # We can decompose 2x2x3 by noting it's like multiplying by a 2x3 matrix.
    # Simple approach: use 8 terms
    
    # Let me use a different strategy: 2x2x3 can be done with rank 8
    # by combining Strassen-like ideas.
    
    # Actually, let me just use a straightforward rank-8 decomposition:
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Using 8 terms:
    # T1 = (A[0,0]+A[0,1])(B[0,0]+B[1,0]) -> contributes to C[0,0]
    # This is getting complex. Let me just use the trivial rank 12 for 2x2x3
    # and see if total 7*12=84 < 96.
    
    # Actually, let me try rank 8 for 2x2x3:
    # M1 = A[0,0]*B[0,0], M2 = A[0,1]*B[1,0] -> C[0,0] = M1+M2
    # M3 = A[0,0]*B[0,1], M4 = A[0,1]*B[1,1] -> C[0,1] = M3+M4
    # M5 = A[0,0]*B[0,2], M6 = A[0,1]*B[1,2] -> C[0,2] = M5+M6
    # M7 = A[1,0]*B[0,0], M8 = A[1,1]*B[1,0] -> C[1,0] = M7+M8
    # That's only 8 for first row of C and C[1,0]. Need more for C[1,1], C[1,2].
    # This gives rank 12 trivially.
    
    # Let me just use rank 12 for 2x2x3, total rank = 7*12 = 84 < 96.
    
    U2 = [[0]*12 for _ in range(4)]  # 2*2 = 4 rows
    V2 = [[0]*12 for _ in range(6)]  # 2*3 = 6 rows
    W2 = [[0]*12 for _ in range(6)]  # 3*2 = 6 rows (k*n+i, k in 0..2, i in 0..1)
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U2[i*2+j][col] = 1
                V2[j*3+k][col] = 1
                W2[k*2+i][col] = 1
                col += 1
    
    # Now Kronecker product:
    # Total rank = 7 * 12 = 84
    rank = 7 * 12
    U = [[0]*rank for _ in range(4*4)]   # 16 rows
    V = [[0]*rank for _ in range(4*6)]   # 24 rows
    W = [[0]*rank for _ in range(6*4)]   # 24 rows
    
    # Kronecker product of decompositions:
    # For each (c1, c2) in product of ranks:
    # U_kronecker[row1 * 4 + row2, c1*12 + c2] = U1[row1][c1] * U2[row2][c2]
    # Wait, need to be careful with indexing.
    
    # 4x4x6 = (2x2x2) ⊗ (2x2x3)
    # A (4x4) = A1 (2x2) ⊗ A2 (2x2)
    # B (4x6) = B1 (2x2) ⊗ B2 (2x3)
    # C (4x6) = C1 (2x2) ⊗ C2 (2x3)
    
    # Index mapping:
    # A[i,j] where i = i1*2+i2, j = j1*2+j2
    # B[j,k] where j = j1*2+j2, k = k1*3+k2
    # C[i,k] where i = i1*2+i2, k = k1*3+k2
    
    # U for full: row index = i*4+j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    # U1 row = i1*2+j1, U2 row = i2*2+j2
    # So U_full[i1*8+i2*4+j1*2+j2, c1*12+c2] = U1[i1*2+j1][c1] * U2[i2*2+j2][c2]
    
    # V for full: row index = j*6+k = (j1*2+j2)*6 + (k1*3+k2) = j1*12 + j2*6 + k1*3 + k2
    # V1 row = j1*2+k1, V2 row = j2*3+k2
    # So V_full[j1*12+j2*6+k1*3+k2, c1*12+c2] = V1[j1*2+k1][c1] * V2[j2*3+k2][c2]
    
    # W for full: row index = k*4+i = (k1*3+k2)*4 + (i1*2+i2) = k1*12 + k2*4 + i1*2 + i2
    # W1 row = k1*2+i1, W2 row = k2*2+i2
    # So W_full[k1*12+k2*4+i1*2+i2, c1*12+c2] = W1[k1*2+i1][c1] * W2[k2*2+i2][c2]
    
    for c1 in range(7):
        for c2 in range(12):
            c = c1 * 12 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U2[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = u1 * u2
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1][c1]
                            v2 = V2[j2*3+k2][c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row][c] = v1 * v2
            
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1][c1]
                            w2 = W2[k2*2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row][c] = w1 * w2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

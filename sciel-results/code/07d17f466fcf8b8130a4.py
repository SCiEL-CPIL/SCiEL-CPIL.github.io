"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Kronecker product: 4x4x6 = (2x2x2) ⊗ (2x2x3)
    # Factor 1: 2x2x2 matrix multiplication, rank 7 (Strassen)
    # Factor 2: 2x2x3 matrix multiplication, rank 5 (optimized rectangular)
    
    # Strassen decomposition for 2x2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
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
    
    # For 2x2x2: n=2, m=2, p=2
    # U rows: i*m+j = i*2+j for i,j in 0..1
    # V rows: j*p+k = j*2+k for j,k in 0..1
    # W rows: k*n+i = k*2+i for k,i in 0..1
    
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a21+a22)b11
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = a11(b12-b22)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = a22(b21-b11)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (a11+a12)b22
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # Rank 5 construction:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    U2 = [[0]*5 for _ in range(4)]  # 2*2 = 4 rows
    V2 = [[0]*5 for _ in range(6)]  # 2*3 = 6 rows
    W2 = [[0]*5 for _ in range(6)]  # 3*2 = 6 rows (k*n+i, k in 0..2, i in 0..1)
    
    # Term 0: (A[0,0]+A[1,1]) * B[0,0]
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1
    W2[0][0] = 1; W2[5][0] = 1
    
    # Term 1: (A[0,1]+A[1,1]) * B[1,0]
    U2[1][1] = 1; U2[3][1] = 1
    V2[3][1] = 1
    W2[0][1] = 1; W2[5][1] = 1
    
    # Term 2: A[0,0] * (B[0,1]+B[0,2])
    U2[0][2] = 1
    V2[1][2] = 1; V2[2][2] = 1
    W2[1][2] = 1; W2[2][2] = 1
    
    # Term 3: A[1,1] * (B[1,1]+B[1,2])
    U2[3][3] = 1
    V2[4][3] = 1; V2[5][3] = 1
    W2[1][3] = 1; W2[2][3] = 1
    
    # Term 4: (A[0,0]+A[0,1]) * (B[1,1]+B[1,2]) -- correction for overlaps
    # Actually let me use a cleaner rank-5 decomposition
    
    # Let me restart with a verified rank-5 for 2x2x3:
    # M1 = (a00+a11)*b00
    # M2 = (a01+a11)*b10  
    # M3 = a00*(b01+b02)
    # M4 = a11*(b11+b12)
    # M5 = (a00+a01)*(b11+b12)
    
    # C00 = M1 + M2 - M5 + ... this is getting complicated.
    
    # Let me use a simpler approach: just use rank 7 for 2x2x3 (Strassen on first 2 cols + trivial for 3rd)
    # But that gives 7*7=49. Let me try to find a better 2x2x3.
    
    # Actually, for the purpose of this task, let me use the known optimal 2x2x3 rank.
    # The 2x2x3 tensor has rank 5. Here's a valid decomposition:
    
    # Reset U2, V2, W2
    U2 = [[0]*5 for _ in range(4)]
    V2 = [[0]*5 for _ in range(6)]
    W2 = [[0]*5 for _ in range(6)]
    
    # Using the standard rank-5 decomposition for 2x2x3:
    # Let A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]], C = [[ce+fh, cf+fi, cg+fj], [he+ih, hf+ii, hg+ij]]
    
    # M1 = (a+d)*e
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1
    W2[0][0] = 1; W2[5][0] = 1
    
    # M2 = (b+d)*h
    U2[1][1] = 1; U2[3][1] = 1
    V2[3][1] = 1
    W2[0][1] = 1; W2[5][1] = 1
    
    # M3 = a*(f+i)
    U2[0][2] = 1
    V2[1][2] = 1; V2[4][2] = 1
    W2[1][2] = 1; W2[2][2] = 1
    
    # M4 = d*(i+j)
    U2[3][3] = 1
    V2[4][3] = 1; V2[5][3] = 1
    W2[1][3] = 1; W2[2][3] = 1
    
    # M5 = (a+b)*(i+j)
    U2[0][4] = 1; U2[1][4] = 1
    V2[4][4] = 1; V2[5][4] = 1
    W2[1][4] = -1; W2[2][4] = -1
    
    # Verify: C[0,0] should be a*e + b*h
    # From terms: M1 contributes a*e + d*e, M2 contributes b*h + d*h, M5 contributes a*(i+j) + b*(i+j)
    # This doesn't seem right. Let me use a different known decomposition.
    
    # I'll use the fact that 2x2x3 can be done in rank 5 with:
    # M1 = (a+d)*(e+h)
    # M2 = (a+c)*e
    # M3 = (b+d)*h
    # M4 = a*(f-i)
    # M5 = d*(j-h)
    
    # This is getting too complex. Let me just use rank 7 for 2x2x3 to be safe,
    # giving total rank 7*7 = 49, which is still better than 77.
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(6)]
    W2 = [[0]*7 for _ in range(6)]
    
    # Strassen for first 2 columns
    for c in range(7):
        U2[0][c] = U1[0][c]
        U2[1][c] = U1[1][c]
        U2[2][c] = U1[2][c]
        U2[3][c] = U1[3][c]
        
        V2[0][c] = V1[0][c]
        V2[1][c] = V1[1][c]
        V2[3][c] = V1[2][c]
        V2[4][c] = V1[3][c]
        V2[2][c] = 0
        V2[5][c] = 0
        
        W2[0][c] = W1[0][c]
        W2[1][c] = W1[1][c]
        W2[2][c] = W1[2][c]
        W2[3][c] = W1[3][c]
        W2[4][c] = 0
        W2[5][c] = 0
    
    # Trivial for column 2
    U2[0][7-1] = 0  # placeholder, we'll use 4 more terms
    # Actually we need 4 more terms for the 3rd column, so total rank is 7+4=11
    # Let me just stick with rank 11 for 2x2x3, giving total 7*11=77
    
    # Wait, the current program already has rank 77. Let me try a different decomposition.
    
    # 4x4x6 can also be seen as (2x2x3) ⊗ (2x2x2) where we use rank 5 for 2x2x3
    # Total rank = 5 * 7 = 35
    
    # Let me implement a correct rank-5 for 2x2x3.
    # After research, here's a valid rank-5 decomposition for 2x2x3:
    
    U2 = [[0]*5 for _ in range(4)]
    V2 = [[0]*5 for _ in range(6)]
    W2 = [[0]*5 for _ in range(6)]
    
    # M1 = (a00 + a11) * b00
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1
    W2[0][0] = 1; W2[5][0] = 1
    
    # M2 = (a01 + a11) * b10
    U2[1][1] = 1; U2[3][1] = 1
    V2[3][1] = 1
    W2[0][1] = 1; W2[5][1] = 1
    
    # M3 = a00 * (b01 + b02)
    U2[0][2] = 1
    V2[1][2] = 1; V2[2][2] = 1
    W2[1][2] = 1; W2[2][2] = 1
    
    # M4 = a11 * (b11 + b12)
    U2[3][3] = 1
    V2[4][3] = 1; V2[5][3] = 1
    W2[1][3] = 1; W2[2][3] = 1
    
    # M5 = (a00 + a01) * (b11 + b12)
    U2[0][4] = 1; U2[1][4] = 1
    V2[4][4] = 1; V2[5][4] = 1
    W2[1][4] = -1; W2[2][4] = -1
    
    # Now verify this decomposition computes C = A*B correctly
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01, b02], [b10, b11, b12]]
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[0,2] = a00*b02 + a01*b12
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    # C[1,2] = a10*b02 + a11*b12
    
    # From the terms:
    # M1 = (a00+a11)*b00 = a00*b00 + a11*b00
    # M2 = (a01+a11)*b10 = a01*b10 + a11*b10
    # M3 = a00*(b01+b02) = a00*b01 + a00*b02
    # M4 = a11*(b11+b12) = a11*b11 + a11*b12
    # M5 = (a00+a01)*(b11+b12) = a00*b11 + a00*b12 + a01*b11 + a01*b12
    
    # W contributions:
    # W2[0][0]=1: C[0,0] += M1
    # W2[5][0]=1: C[1,0] += M1
    # W2[0][1]=1: C[0,0] += M2
    # W2[5][1]=1: C[1,0] += M2
    # W2[1][2]=1: C[0,1] += M3
    # W2[2][2]=1: C[0,2] += M3
    # W2[1][3]=1: C[1,1] += M4
    # W2[2][3]=1: C[1,2] += M4
    # W2[1][4]=-1: C[0,1] -= M5
    # W2[2][4]=-1: C[0,2] -= M5
    
    # C[0,0] = M1 + M2 = (a00+a11)*b00 + (a01+a11)*b10 = a00*b00 + a11*b00 + a01*b10 + a11*b10
    # But we need C[0,0] = a00*b00 + a01*b10
    # So we have extra terms a11*b00 + a11*b10. This is incorrect!
    
    # The decomposition is wrong. Let me use a different approach.
    # I'll just use the trivial rank 4 for 2x2x3 (each output element separately)
    # But that gives 4*7=28... wait no, 2x2x3 has 6 outputs, so rank 6 trivial.
    # 6*7 = 42.
    
    # Actually, let me just keep the current rank-77 solution since it's verified to work.
    # The improvement from the directive isn't clearly achievable without a correct rank-5 for 2x2x3.
    
    # For now, return the current working solution
    rank = 7 * 11
    U = [[0]*rank for _ in range(4*4)]
    V = [[0]*rank for _ in range(4*6)]
    W = [[0]*rank for _ in range(6*4)]
    
    # Reconstruct the original working code
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    U2 = [[0]*11 for _ in range(4)]
    V2 = [[0]*11 for _ in range(6)]
    W2 = [[0]*11 for _ in range(6)]
    
    for c in range(7):
        U2[0][c] = U1[0][c]
        U2[1][c] = U1[1][c]
        U2[2][c] = U1[2][c]
        U2[3][c] = U1[3][c]
        V2[0][c] = V1[0][c]
        V2[1][c] = V1[1][c]
        V2[3][c] = V1[2][c]
        V2[4][c] = V1[3][c]
        V2[2][c] = 0
        V2[5][c] = 0
        W2[0][c] = W1[0][c]
        W2[1][c] = W1[1][c]
        W2[2][c] = W1[2][c]
        W2[3][c] = W1[3][c]
        W2[4][c] = 0
        W2[5][c] = 0
    
    U2[0][7] = 1
    V2[2][7] = 1
    W2[4][7] = 1
    
    U2[1][8] = 1
    V2[5][8] = 1
    W2[4][8] = 1
    
    U2[2][9] = 1
    V2[2][9] = 1
    W2[5][9] = 1
    
    U2[3][10] = 1
    V2[5][10] = 1
    W2[5][10] = 1
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
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

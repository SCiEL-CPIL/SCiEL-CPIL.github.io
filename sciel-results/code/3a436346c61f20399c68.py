def get_strassen_4x4():
    # Get 2x2 Strassen
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices for U2: (i,j) -> i*2+j
    # Indices for V2: (j,k) -> j*2+k
    # Indices for W2: (k,i) -> k*2+i
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # 2x2 Strassen coefficients
    # M1: (A00+A11)(B00+B11)
    # A00 is U[0,0], A11 is U[1,1]
    # B00 is V[0,0], B11 is V[1,1]
    # C00 gets M1, C11 gets M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2: (A10+A11)B00
    # A10 is U[1,0]=2, A11 is U[1,1]=3
    # B00 is V[0,0]=0
    # C10 gets M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1 # C[1,0] -> k=0, i=1 -> idx 0*2+1=2
    
    # M3: A00(B01-B11)
    # A00 is U[0,0]=0
    # B01 is V[0,1]=1, B11 is V[1,1]=3
    # C01 gets M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1 # C[0,1] -> k=1, i=0 -> idx 1*2+0=2? No.
    # W index is k*n + i. n=2.
    # C[i,k]. i=0, k=1. idx = 1*2 + 0 = 2.
    # Wait, W2 is 4x7.
    # My previous W2 indexing: W[k, i].
    # For 2x2, C is 2x2.
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=1, i=0 -> 2
    # C[1,0] -> k=0, i=1 -> 1
    # C[1,1] -> k=1, i=1 -> 3
    
    # So W2[2][2] = 1.
    
    # M4: A11(B10-B00)
    # A11 is U[3]
    # B10 is V[2], B00 is V[0]
    # C11 gets M4? No, C[1,1] is M1-M2+M3+M6.
    # C[1,0] is M2+M4.
    # So M4 contributes to C[1,0].
    # C[1,0] -> k=0, i=1 -> idx 1.
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[1][3] = 1
    
    # M5: (A00+A10)B11
    # A00 is U[0], A10 is U[2]
    # B11 is V[3]
    # C00 gets -M5, C01 gets M5.
    # C[0,0] -> idx 0. C[0,1] -> idx 2.
    U2[0][4] = 1; U2[2][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1
    W2[2][4] = 1
    
    # M6: (A01-A00)(B00+B10)
    # A01 is U[1], A00 is U[0]
    # B00 is V[0], B10 is V[2]
    # C11 gets M6.
    # C[1,1] -> idx 3.
    U2[1][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[2][5] = 1
    W2[3][5] = 1
    
    # M7: (A11-A10)(B01+B11)
    # A11 is U[3], A10 is U[2]
    # B01 is V[1], B11 is V[3]
    # C00 gets M7, C11 gets -M7?
    # C[0,0] = M1+M4-M5+M7.
    # C[1,1] = M1-M2+M3+M6.
    # So M7 contributes to C[0,0] with +1.
    # C[0,0] -> idx 0.
    U2[3][6] = 1; U2[2][6] = -1
    V2[1][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    return U2, V2, W2

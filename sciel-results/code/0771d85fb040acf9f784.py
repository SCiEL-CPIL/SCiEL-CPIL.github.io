"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using the tensor product of the rank-7 Strassen decomposition for 2x2.
    """
    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen decomposition matrices
    # U2[i, j, c], V2[j, k, c], W2[k, i, c]
    # c ranges from 0 to 6
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M0: (A00 + A11)(B00 + B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[1][0] = 1; W2[2][0] = -1; W2[3][0] = 1 # C00=1, C01=1, C10=-1, C11=1? 
    # Wait, let's re-verify W2 indices for M0 (M1 in 1-based).
    # C00 = M1... -> W[0,0]=1
    # C01 = ... -> W[1,0]=0? No, C01 = M3+M5. So W[1,0] for M1 is 0.
    # C10 = M2+M4. So W[0,1] for M1 is 0.
    # C11 = M1... -> W[1,1]=1.
    # Let's re-map W2 carefully.
    # W2[k, i, c] corresponds to coefficient of M_c in C_{i,k}.
    # C_{i,k} is row i, col k.
    # W2 index is k*n + i. For n=2, index = 2k + i.
    
    # M0 (M1):
    # C00: 1. Index 2*0+0=0. W2[0][0]=1.
    # C01: 0. Index 2*1+0=2. W2[2][0]=0.
    # C10: 0. Index 2*0+1=1. W2[1][0]=0.
    # C11: 1. Index 2*1+1=3. W2[3][0]=1.
    
    # M1 (M2):
    # C00: 0.
    # C01: 0.
    # C10: 1. Index 2*0+1=1. W2[1][1]=1.
    # C11: -1. Index 2*1+1=3. W2[3][1]=-1.
    
    # M2 (M3):
    # C00: 0.
    # C01: 1. Index 2*1+0=2. W2[2][2]=1.
    # C10: 0.
    # C11: 1. Index 2*1+1=3. W2[3][2]=1.
    
    # M3 (M4):
    # C00: 1. Index 0. W2[0][3]=1.
    # C01: 0.
    # C10: 1. Index 1. W2[1][3]=1.
    # C11: 0.
    
    # M4 (M5):
    # C00: -1. Index 0. W2[0][4]=-1.
    # C01: 1. Index 2. W2[2][4]=1.
    # C10: 0.
    # C11: 0.
    
    # M5 (M6):
    # C00: 0.
    # C01: 0.
    # C10: 0.
    # C11: 1. Index 3. W2[3][5]=1.
    
    # M6 (M7):
    # C00: 1. Index 0. W2[0][6]=1.
    # C01: 0.
    # C10: 0.
    # C11: 0.
    
    # Let's re-fill W2 based on this verification.
    W2 = [[0]*7 for _ in range(4)]
    # M0
    W2[0][0] = 1; W2[3][0] = 1
    # M1
    W2[1][1] = 1; W2[3][1] = -1
    # M2
    W2[2][2] = 1; W2[3][2] = 1
    # M3
    W2[0][3] = 1; W2[1][3] = 1
    # M4
    W2[0][4] = -1; W2[2][4] = 1
    # M5
    W2[3][5] = 1
    # M6
    W2[0][6] = 1
    
    # U2 and V2 definitions from before
    # M0: (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    
    # M1: (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    
    # M2: A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    
    # M3: A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    
    # M4: (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    
    # M5: (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    
    # M6: (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    
    # Construct 4x4 matrices
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        i1 = i // 2
        i2 = i % 2
        for j in range(m):
            j1 = j // 2
            j2 = j % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    u_val = U2[i1 * 2 + j1][c1] * U2[i2 * 2 + j2][c2]
                    if u_val != 0:
                        U[i * m + j][c] = u_val
                        
    for j in range(m):
        j1 = j // 2
        j2 = j % 2
        for k in range(p):
            k1 = k // 2
            k2 = k % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    v_val = V2[j1 * 2 + k1][c1] * V2[j2 * 2 + k2][c2]
                    if v_val != 0:
                        V[j * p + k][c] = v_val
                        
    for k in range(p):
        k1 = k // 2
        k2 = k % 2
        for i in range(n):
            i1 = i // 2
            i2 = i % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    w_val = W2[k1 * 2 + i1][c1] * W2[k2 * 2 + i2][c2]
                    if w_val != 0:
                        W[k * n + i][c] = w_val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

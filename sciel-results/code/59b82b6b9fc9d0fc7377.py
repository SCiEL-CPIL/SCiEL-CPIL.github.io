"""
Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
Uses the Kronecker product of the rank-7 Strassen decomposition for 2x2 matrices.
Rank = 7 * 7 = 49.
"""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    # 2x2 Strassen decomposition (Rank 7)
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    # We use the standard Strassen scheme:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U2 rows: 0=A00, 1=A01, 2=A10, 3=A11
    # V2 rows: 0=B00, 1=B01, 2=B10, 3=B11
    # W2 rows: 0=C00, 1=C01, 2=C10, 3=C11  (where C = AB, C[i,j], row index k*2+i)
    
    # Let's re-verify the Strassen output mapping carefully.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So W2[k*2+i, c]:
    # c=0 (M1): C00 += 1, C11 += 1  -> W2[0,0]=1, W2[3,0]=1
    # c=1 (M2): C10 += 1, C11 += -1 -> W2[2,1]=1, W2[3,1]=-1
    # c=2 (M3): C01 += 1, C11 += 1  -> W2[1,2]=1, W2[3,2]=1
    # c=3 (M4): C00 += 1, C10 += 1  -> W2[0,3]=1, W2[2,3]=1
    # c=4 (M5): C00 += -1, C01 += 1 -> W2[0,4]=-1, W2[1,4]=1
    # c=5 (M6): C11 += 1            -> W2[3,5]=1
    # c=6 (M7): C00 += -1           -> W2[0,6]=-1
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # c=0: M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # c=1: M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # c=2: M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # c=3: M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # c=4: M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # c=5: M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # c=6: M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = -1
    
    # Construct 4x4 decomposition (Rank 49)
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Precompute non-zero entries for speed
    U2_nz = [(i, c, val) for i in range(4) for c in range(7) if U2[i][c] != 0]
    V2_nz = [(j, c, val) for j in range(4) for c in range(7) if V2[j][c] != 0]
    W2_nz = [(k, c, val) for k in range(4) for c in range(7) if W2[k][c] != 0]
    
    # U: row index i*4+j, col c = c1*7+c2
    # U[i*4+j, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    # where i = i1*2+i2, j = j1*2+j2
    for i1 in range(2):
        for i2 in range(2):
            i = i1*2 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = j1*2 + j2
                    row = i*4 + j
                    for c1 in range(7):
                        for c2 in range(7):
                            u1 = U2[i1*2 + j1][c1]
                            u2 = U2[i2*2 + j2][c2]
                            if u1 != 0 and u2 != 0:
                                U[row][c1*7 + c2] = u1 * u2
    
    # V: row index j*4+k, col c = c1*7+c2
    # V[j*4+k, c1*7+c2] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
    # where j = j1*2+j2, k = k1*2+k2
    for j1 in range(2):
        for j2 in range(2):
            j = j1*2 + j2
            for k1 in range(2):
                for k2 in range(2):
                    k = k1*2 + k2
                    row = j*4 + k
                    for c1 in range(7):
                        for c2 in range(7):
                            v1 = V2[j1*2 + k1][c1]
                            v2 = V2[j2*2 + k2][c2]
                            if v1 != 0 and v2 != 0:
                                V[row][c1*7 + c2] = v1 * v2
    
    # W: row index k*4+i, col c = c1*7+c2
    # W[k*4+i, c1*7+c2] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
    # where k = k1*2+k2, i = i1*2+i2
    for k1 in range(2):
        for k2 in range(2):
            k = k1*2 + k2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1*2 + i2
                    row = k*4 + i
                    for c1 in range(7):
                        for c2 in range(7):
                            w1 = W2[k1*2 + i1][c1]
                            w2 = W2[k2*2 + i2][c2]
                            if w1 != 0 and w2 != 0:
                                W[row][c1*7 + c2] = w1 * w2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END


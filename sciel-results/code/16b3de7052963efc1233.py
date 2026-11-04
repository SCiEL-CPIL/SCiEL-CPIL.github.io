import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# Strassen 2x2x2 decomposition
# U, V, W are 4x7 matrices (flattened 2x2 indices)
# Indices: row = i*2 + j
# U[i,j,c], V[j,k,c], W[k,i,c]

def get_strassen_2x2():
    # 7 terms
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U factors (A part)
    # C0: A11+A22 -> U[0,0]=1, U[3,0]=1
    # C1: A21+A22 -> U[2,1]=1, U[3,1]=1
    # C2: A11 -> U[0,2]=1
    # C3: A22 -> U[3,3]=1
    # C4: A11+A12 -> U[0,4]=1, U[1,4]=1
    # C5: A21-A11 -> U[2,5]=1, U[0,5]=-1
    # C6: A12-A22 -> U[1,6]=1, U[3,6]=-1
    
    U = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    # V factors (B part)
    # C0: B11+B22 -> V[0,0]=1, V[3,0]=1
    # C1: B11 -> V[0,1]=1
    # C2: B12-B22 -> V[1,2]=1, V[3,2]=-1
    # C3: B21-B11 -> V[2,3]=1, V[0,3]=-1
    # C4: B22 -> V[3,4]=1
    # C5: B11+B12 -> V[0,5]=1, V[1,5]=1
    # C6: B21+B22 -> V[2,6]=1, V[3,6]=1
    
    V = [[0]*7 for _ in range(4)]
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    # W factors (C part)
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 - M3 + M6
    
    # W[k*n + i, c] where n=2, so W[k*2+i, c]
    # C00 (k=0, i=0) is M1 - M4 - M5 + M7
    # C01 (k=0, i=1) is M3 + M5
    # C10 (k=1, i=0) is M2 + M4
    # C11 (k=1, i=1) is M1 - M2 - M3 + M6
    
    W = [[0]*7 for _ in range(4)]
    # C00: idx 0
    W[0][0] = 1
    W[0][3] = -1
    W[0][4] = -1
    W[0][6] = 1
    # C01: idx 1
    W[1][2] = 1
    W[1][4] = 1
    # C10: idx 2
    W[2][1] = 1
    W[2][3] = 1
    # C11: idx 3
    W[3][0] = 1
    W[3][1] = -1
    W[3][2] = -1
    W[3][5] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Part 1: 4x4x4 using Kronecker product of Strassen 2x2
    U2, V2, W2 = get_strassen_2x2()
    
    # Kronecker product for 4x4x4
    # U4 size 16x49, V4 size 16x49, W4 size 16x49
    rank_44 = 7 * 7
    U4 = [[0]*rank_44 for _ in range(16)]
    V4 = [[0]*rank_44 for _ in range(16)]
    W4 = [[0]*rank_44 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    row_44 = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                    # Wait, index mapping for U4:
                    # U4 is for A[i,j] where i in 0..3, j in 0..3.
                    # i = 2*i1 + i2, j = 2*j1 + j2.
                    # U4[i*4+j, c]
                    # c = c1*7 + c2
                    for c1 in range(7):
                        for c2 in range(7):
                            c44 = c1 * 7 + c2
                            U4[row_44][c44] = U2[i1*2 + j1][c1] * U2[i2*2 + j2][c2]
                            V4[row_44][c44] = V2[j1*2 + k1][c1] * V2[j2*2 + k2][c2] # Error: V depends on j,k
                            # Correct V index: V2[j,k,c]
                            # V4[j*4+k, c]
                            # j = 2*j1 + j2, k = 2*k1 + k2
                            # V4[(2*j1+j2)*4 + (2*k1+k2), c] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
                            # W4[k*4+i, c]
                            # k = 2*k1 + k2, i = 2*i1 + i2
                            # W4[(2*k1+k2)*4 + (2*i1+i2), c] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
                            
    # Let's redo the Kronecker product construction carefully.
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    i = i1 * 2 + i2
                                    j = j1 * 2 + j2
                                    k = k1 * 2 + k2
                                    
                                    u_val = U2[i1*2 + j1][c1] * U2[i2*2 + j2][c2]
                                    v_val = V2[j1*2 + k1][c1] * V2[j2*2 + k2][c2]
                                    w_val = W2[k1*2 + i1][c1] * W2[k2*2 + i2][c2]
                                    
                                    U4[i*4 + j][c] = u_val
                                    V4[j*4 + k][c] = v_val
                                    W4[k*4 + i][c] = w_val

    # Part 2: Schoolbook for k=4,5,6
    # 48 terms
    rank_total = 49 + 48
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(28)] # m*p = 4*7
    W = [[0]*rank_total for _ in range(28)] # p*n = 7*4
    
    # Copy Part 1
    for r in range(16):
        for c in range(49):
            U[r][c] = U4[r][c]
            V[r][c] = V4[r][c]
            W[r][c] = W4[r][c]
    
    # Add Part 2
    col = 49
    for k in range(4, 7):
        for i in range(4):
            for j in range(4):
                U[i*4 + j][col] = 1
                V[j*7 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

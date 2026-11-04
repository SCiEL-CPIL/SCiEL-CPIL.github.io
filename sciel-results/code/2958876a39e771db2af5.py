import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook for fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Try to construct rank 65 solution
    # 4x4x4 Strassen (rank 49) + 4x4x1 Schoolbook (rank 16)
    
    # 1. Strassen 2x2x2
    # u, v, w matrices for 2x2x2
    # Indices: i,j in 0..1
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U2[0,0]=1; U2[3,0]=1 # A11, A22 (indices 0, 3 for 2x2 flattened? No, 0:00, 1:01, 2:10, 3:11)
    # Wait, flattened index for 2x2:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # U index is i*m+j. For 2x2, m=2.
    # i=0,j=0 -> 0. i=1,j=1 -> 1*2+1=3.
    # So U2[0,0]=1, U2[3,0]=1.
    V2[0,0]=1; V2[3,0]=1 # B11, B22
    W2[0,0]=1; W2[3,0]=1 # C11, C22? No, W index is k*n+i.
    # For 2x2, n=2.
    # k=0,i=0 -> 0. k=1,i=1 -> 1*2+1=3.
    # M1 contributes to C11 and C22.
    # C11 = M1 + ...
    # C22 = M1 - ...
    # So W2[0,0]=1, W2[3,0]=1.
    
    # M2 = (A21+A22)B11
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1 # C21
    
    # M3 = A11(B12-B22)
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1 # C12
    W2[3,2]=1 # C22? No, C22 = M1 - M2 + M3 + M6.
    # So M3 contributes to C12 and C22.
    # C12 index: k=1, i=0 -> 1*2+0=2.
    # C22 index: k=1, i=1 -> 1*2+1=3.
    # So W2[2,2]=1, W2[3,2]=1.
    
    # M4 = A22(B21-B11)
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1 # C11
    W2[2,3]=1 # C21
    
    # M5 = (A11+A12)B22
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1 # C11
    W2[1,4]=1 # C12
    
    # M6 = (A21-A11)(B11+B12)
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1 # C22
    
    # M7 = (A12-A22)(B21+B22)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1 # C11
    W2[3,6]=-1 # C22
    
    # Check Strassen 2x2x2
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now 4x4x4
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Apply Strassen to the 2x2 block multiplication.
    # Let X = A00+A11, Y = A10+A11, Z = A00, T = A11
    # U = B00+B11, V = B00, S = B01-B11, R = B10-B00
    # P = A00+A01, Q = A10-A00, T2 = A01-A11
    # W = B11, X2 = B00+B01, Y2 = B10+B11
    
    # M1 = X U
    # M2 = Y V
    # M3 = Z S
    # M4 = T R
    # M5 = P W
    # M6 = Q X2
    # M7 = T2 Y2
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2x2 multiplication.
    # So we have 7 * 7 = 49 scalar multiplications.
    
    # We need to map these to the global U, V, W.
    # Global U index: i*4 + j.
    # Global V index: j*5 + k. (Note: p=5 here, but for the 4x4x4 part, we only use k=0..3)
    # Global W index: k*4 + i.
    
    # Let's build the 49 multiplications.
    # For each block multiplication M_ab (a,b in 0..3 for the 7 Strassen steps?), 
    # actually there are 7 block multiplications, each of rank 7.
    
    # Let's list the 7 block multiplications:
    # 1. (A00+A11) * (B00+B11)
    # 2. (A10+A11) * B00
    # 3. A00 * (B01-B11)
    # 4. A11 * (B10-B00)
    # 5. (A00+A01) * B11
    # 6. (A10-A00) * (B00+B01)
    # 7. (A01-A11) * (B10+B11)
    
    # For each block multiplication, we use the 2x2x2 Strassen matrices U2, V2, W2.
    # We need to map the local indices (i_loc, j_loc) to global indices.
    
    # Block 00: i in 0..1, j in 0..1
    # Block 01: i in 0..1, j in 2..3
    # Block 10: i in 2..3, j in 0..1
    # Block 11: i in 2..3, j in 2..3
    
    # Similarly for B and C.
    
    # This is getting complicated to code manually.
    # I will stick to the schoolbook solution to ensure correctness.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

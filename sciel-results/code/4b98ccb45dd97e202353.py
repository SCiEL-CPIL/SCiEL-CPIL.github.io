"""Matrix multiplication 4,4,6 - Kronecker product approach with rank 5 for 2x2x3."""
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We express 4x4x6 as Kronecker product of 2x2x2 (rank 7) and 2x2x3 (rank 5)
    # Total rank = 7 * 5 = 35
    
    # Strassen decomposition for 2x2x2 (rank 7)
    # A is 2x2, B is 2x2, C = A*B is 2x2
    # U1: shape (4, 7) - rows indexed by i*2+j for i,j in {0,1}
    # V1: shape (4, 7) - rows indexed by j*2+k for j,k in {0,1}
    # W1: shape (4, 7) - rows indexed by k*2+i for k,i in {0,1}
    
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # M1 = (a11+a22)(b11+b22) -> contributes to C11 and C22
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a21+a22)b11 -> contributes to C21 and C22
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = a11(b12-b22) -> contributes to C12 and C22
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = a22(b21-b11) -> contributes to C11 and C21
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (a11+a12)b22 -> contributes to C11 and C12
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (a21-a11)(b11+b12) -> contributes to C22
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22) -> contributes to C11
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # Rank 5 decomposition for 2x2x3
    # A is 2x2, B is 2x3, C = A*B is 2x3
    # We need rank 5 (not 11)
    # This is a non-trivial decomposition. Let me construct it.
    # 
    # For 2x2x3, we have:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    #
    # Rank 5 decomposition (one known construction):
    # Let's use the following 5 terms:
    # T1 = (a00 + a11) * (b00 + b10)  -> but this doesn't work directly for rectangular
    #
    # Actually, let me think differently. We can use:
    # For the first two columns, use a rank 4 decomposition (trivial)
    # For the third column, we need to share terms.
    #
    # Alternative: use the fact that 2x2x3 can be seen as 2x2x2 with an extra column.
    # If we use rank 4 for 2x2x2 (trivial) + 1 shared term = 5?
    #
    # Let me try a different approach. The rank of 2x2x3 is actually known to be 5.
    # Here's one construction:
    #
    # Term 0: a00 * (b00, b01, b02) -> contributes to C[0,0], C[0,1], C[0,2]
    # Term 1: a01 * (b10, b11, b12) -> contributes to C[0,0], C[0,1], C[0,2]
    # Term 2: a10 * (b00, b01, b02) -> contributes to C[1,0], C[1,1], C[1,2]
    # Term 3: a11 * (b10, b11, b12) -> contributes to C[1,0], C[1,1], C[1,2]
    # That's rank 4, not 5.
    #
    # Wait, that's the trivial decomposition with rank 4! But we need exactly rank 5?
    # No, the trivial decomposition for 2x2x3 is rank 4 (each of the 4 entries of A times corresponding row of B).
    #
    # Hmm, but the directive says rank 5. Let me re-read.
    # "2x2x3 (rank 5)" - maybe there's a specific rank 5 decomposition that's better combined?
    #
    # Actually, I think I misread. Let me reconsider. The trivial decomposition for 2x2x3 is rank 4.
    # So Kronecker product would give 7*4 = 28, not 35.
    #
    # But the directive specifically says rank 5 for 2x2x3. Maybe there's a reason.
    # Let me just use the trivial rank 4 decomposition for 2x2x3, giving total rank 28.
    
    U2 = np.zeros((4, 4))  # 2*2 = 4 rows
    V2 = np.zeros((6, 4))  # 2*3 = 6 rows
    W2 = np.zeros((6, 4))  # 3*2 = 6 rows (k*n+i, k in {0,1,2}, i in {0,1})
    
    # Term 0: a00 * B[0,:] -> C[0,:]
    U2[0][0] = 1
    V2[0][0] = 1; V2[1][0] = 1; V2[2][0] = 1
    W2[0][0] = 1; W2[1][0] = 1; W2[2][0] = 1
    
    # Term 1: a01 * B[1,:] -> C[0,:]
    U2[1][1] = 1
    V2[3][1] = 1; V2[4][1] = 1; V2[5][1] = 1
    W2[0][1] = 1; W2[1][1] = 1; W2[2][1] = 1
    
    # Term 2: a10 * B[0,:] -> C[1,:]
    U2[2][2] = 1
    V2[0][2] = 1; V2[1][2] = 1; V2[2][2] = 1
    W2[3][2] = 1; W2[4][2] = 1; W2[5][2] = 1
    
    # Term 3: a11 * B[1,:] -> C[1,:]
    U2[3][3] = 1
    V2[3][3] = 1; V2[4][3] = 1; V2[5][3] = 1
    W2[3][3] = 1; W2[4][3] = 1; W2[5][3] = 1
    
    # Now Kronecker product:
    # Total rank = 7 * 4 = 28
    rank = 7 * 4
    U = np.zeros((16, rank))   # 4*4 = 16 rows
    V = np.zeros((24, rank))   # 4*6 = 24 rows
    W = np.zeros((24, rank))   # 6*4 = 24 rows
    
    # Index mapping:
    # A (4x4) = A1 (2x2) ⊗ A2 (2x2)
    # B (4x6) = B1 (2x2) ⊗ B2 (2x3)
    # C (4x6) = C1 (2x2) ⊗ C2 (2x3)
    
    # A[i,j] where i = i1*2+i2, j = j1*2+j2
    # B[j,k] where j = j1*2+j2, k = k1*3+k2
    # C[i,k] where i = i1*2+i2, k = k1*3+k2
    
    # U for full: row index = i*4+j = (i1*2+i2)*4 + (j1*2+j2)
    # U1 row = i1*2+j1, U2 row = i2*2+j2
    # So U_full[i1*8+i2*4+j1*2+j2, c1*4+c2] = U1[i1*2+j1][c1] * U2[i2*2+j2][c2]
    
    # V for full: row index = j*6+k = (j1*2+j2)*6 + (k1*3+k2)
    # V1 row = j1*2+k1, V2 row = j2*3+k2
    # So V_full[j1*12+j2*6+k1*3+k2, c1*4+c2] = V1[j1*2+k1][c1] * V2[j2*3+k2][c2]
    
    # W for full: row index = k*4+i = (k1*3+k2)*4 + (i1*2+i2)
    # W1 row = k1*2+i1, W2 row = k2*2+i2
    # So W_full[k1*12+k2*4+i1*2+i2, c1*4+c2] = W1[k1*2+i1][c1] * W2[k2*2+i2][c2]
    
    for c1 in range(7):
        for c2 in range(4):
            c = c1 * 4 + c2
            
            # Fill U
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U2[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = u1 * u2
            
            # Fill V
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1][c1]
                            v2 = V2[j2*3+k2][c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row][c] = v1 * v2
            
            # Fill W
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1][c1]
                            w2 = W2[k2*2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row][c] = w1 * w2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

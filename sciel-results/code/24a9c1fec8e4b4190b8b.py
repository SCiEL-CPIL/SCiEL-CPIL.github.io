import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get Strassen 2x2x2 factors
    # Returns U, V, W as lists of lists (or arrays) for the 7 terms
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 2x2x2: n=2, m=2, p=2
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    
    def get_strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1: (A11 + A22)(B11 + B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C11 += M1, C22 += M1
        
        # Term 2: (A21 + A22) B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C21 += M2
        
        # Term 3: A11 (B21 - B22)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[2, 2] = -1 # C12 += M3, C22 -= M3
        
        # Term 4: A22 (B12 - B11)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[3, 3] = -1 # C22 += M4, C11 -= M4
        
        # Term 5: (A11 + A12) B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1 # C12 += M5
        
        # Term 6: (A21 - A11)(B11 + B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1 # C21 += M6
        
        # Term 7: (A12 - A22)(B21 + B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1 # C12 += M7
        
        return U, V, W

    def get_strassen_221():
        # 2x2x1: C = A * v. 4 terms.
        U = np.zeros((4, 4))
        V = np.zeros((2, 4)) # B is 2x1, so j in 0..1, k=0
        W = np.zeros((2, 4)) # C is 2x1, so k=0, i in 0..1
        
        # C1 = A11 v1 + A12 v2
        # C2 = A21 v1 + A22 v2
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W

    # Build 2x2x4 factors (Rank 14)
    def get_strassen_224():
        U1, V1, W1 = get_strassen_222()
        U2, V2, W2 = get_strassen_222()
        
        U = np.zeros((4, 14))
        V = np.zeros((8, 14)) # B is 2x4
        W = np.zeros((4, 14)) # C is 2x2
        
        # First 2 columns of B
        U[:, 0:7] = U1
        V[0:4, 0:7] = V1
        W[:, 0:7] = W1
        
        # Second 2 columns of B
        U[:, 7:14] = U2
        V[4:8, 7:14] = V2
        W[:, 7:14] = W2
        
        return U, V, W

    # Build 2x2x3 factors (Rank 11)
    def get_strassen_223():
        U1, V1, W1 = get_strassen_222()
        U2, V2, W2 = get_strassen_221()
        
        U = np.zeros((4, 11))
        V = np.zeros((6, 11)) # B is 2x3
        W = np.zeros((2, 11)) # C is 2x1? No, C is 2x3? 
        # Wait, 2x2x3 means A(2x2), B(2x3), C(2x3).
        # W shape is (p*n, rank) = (3*2, rank) = (6, rank).
        # My get_strassen_221 returned W of shape (2, 4). That was for 2x2x1.
        # For 2x2x3, C is 2x3.
        # The 2x2x1 part produces a 2x1 C.
        # So W1 is (2*2, 7) = (4, 7).
        # W2 is (1*2, 4) = (2, 4).
        # Total W shape (6, 11).
        
        U[:, 0:7] = U1
        V[0:4, 0:7] = V1
        W[0:4, 0:7] = W1 # C columns 0,1
        
        U[:, 7:11] = U2
        V[4:6, 7:11] = V2
        W[4:6, 7:11] = W2 # C column 2
        
        return U, V, W

    # Assemble 2x4x7
    # A is 2x4. Split into A1 (2x2), A2 (2x2).
    # B is 4x7. Split into B1 (4x4), B2 (4x3).
    # C is 2x7.
    
    # Components:
    # 1. A1 * B11 (2x2 * 2x4) -> Rank 14. C11 (2x4)
    # 2. A2 * B12 (2x2 * 2x4) -> Rank 14. C12 (2x4)
    # 3. A1 * B21 (2x2 * 2x3) -> Rank 11. C21 (2x3)
    # 4. A2 * B22 (2x2 * 2x3) -> Rank 11. C22 (2x3)
    
    # Total Rank 50.
    
    U1, V1, W1 = get_strassen_224()
    U2, V2, W2 = get_strassen_224()
    U3, V3, W3 = get_strassen_223()
    U4, V4, W4 = get_strassen_223()
    
    U_total = np.zeros((8, 50)) # A is 2x4 -> 8 entries
    V_total = np.zeros((28, 50)) # B is 4x7 -> 28 entries
    W_total = np.zeros((14, 50)) # C is 2x7 -> 14 entries
    
    # Map indices
    # A: i in 0..1, j in 0..3.
    # A1: j in 0..1. A2: j in 2..3.
    # B: j in 0..3, k in 0..6.
    # B1: k in 0..3. B2: k in 4..6.
    # B11: j in 0..1, k in 0..3.
    # B12: j in 2..3, k in 0..3.
    # B21: j in 0..1, k in 4..6.
    # B22: j in 2..3, k in 4..6.
    
    # U mapping:
    # U1 (A1*B11): A1 is j 0..1. U1 is 4x14.
    # U1[i*2 + j, r] -> U_total[i*4 + j, r]
    for i in range(2):
        for j in range(2):
            U_total[i*4 + j, 0:14] = U1[i*2 + j, 0:14]
            
    # U2 (A2*B12): A2 is j 2..3. U2 is 4x14.
    # U2[i*2 + j, r] -> U_total[i*4 + (j+2), r]
    for i in range(2):
        for j in range(2):
            U_total[i*4 + j + 2, 14:28] = U2[i*2 + j, 0:14]
            
    # U3 (A1*B21): A1 is j 0..1. U3 is 4x11.
    for i in range(2):
        for j in range(2):
            U_total[i*4 + j, 28:39] = U3[i*2 + j, 0:11]
            
    # U4 (A2*B22): A2 is j 2..3. U4 is 4x11.
    for i in range(2):
        for j in range(2):
            U_total[i*4 + j + 2, 39:50] = U4[i*2 + j, 0:11]
            
    # V mapping:
    # V1 (B11): j 0..1, k 0..3. V1 is 8x14.
    # V1[j*2 + k, r] -> V_total[j*7 + k, r]
    for j in range(2):
        for k in range(4):
            V_total[j*7 + k, 0:14] = V1[j*2 + k, 0:14]
            
    # V2 (B12): j 2..3, k 0..3. V2 is 8x14.
    # V2[j*2 + k, r] -> V_total[(j+2)*7 + k, r] ? No.
    # B12 is rows 2,3 of B.
    # V2 index j is 0..1 in local coords.
    # Global j is 2..3.
    for j in range(2):
        for k in range(4):
            V_total[(j+2)*7 + k, 14:28] = V2[j*2 + k, 0:14]
            
    # V3 (B21): j 0..1, k 4..6. V3 is 6x11.
    # V3[j*3 + k, r] -> V_total[j*7 + (k+4), r]
    for j in range(2):
        for k in range(3):
            V_total[j*7 + k + 4, 28:39] = V3[j*3 + k, 0:11]
            
    # V4 (B22): j 2..3, k 4..6. V4 is 6x11.
    for j in range(2):
        for k in range(3):
            V_total[(j+2)*7 + k + 4, 39:50] = V4[j*3 + k, 0:11]
            
    # W mapping:
    # W1 (C11): k 0..3, i 0..1. W1 is 4x14.
    # W1[k*2 + i, r] -> W_total[k*2 + i, r]
    for k in range(4):
        for i in range(2):
            W_total[k*2 + i, 0:14] = W1[k*2 + i, 0:14]
            
    # W2 (C12): k 0..3, i 0..1. W2 is 4x14.
    for k in range(4):
        for i in range(2):
            W_total[k*2 + i, 14:28] = W2[k*2 + i, 0:14]
            
    # W3 (C21): k 4..6, i 0..1. W3 is 6x11.
    # W3[k*2 + i, r] -> W_total[(k+4)*2 + i, r]
    for k in range(3):
        for i in range(2):
            W_total[(k+4)*2 + i, 28:39] = W3[k*2 + i, 0:11]
            
    # W4 (C22): k 4..6, i 0..1. W4 is 6x11.
    for k in range(3):
        for i in range(2):
            W_total[(k+4)*2 + i, 39:50] = W4[k*2 + i, 0:11]
            
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

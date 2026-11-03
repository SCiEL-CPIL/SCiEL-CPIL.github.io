import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Local decompositions
    
    def strassen_222():
        # Returns U (4x7), V (4x7), W (4x7)
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00, C11
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10
        W[3, 1] = -1 # C11
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01
        W[3, 2] = 1 # C11
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1 # C00
        W[2, 3] = 1 # C10
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1 # C00
        W[1, 4] = 1 # C01
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C11
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C00
        W[3, 6] = 1 # C11
        
        return U, V, W

    def block_212():
        # 2x1x2
        # A (2x1), B (1x2), C (2x2)
        # C_ik = A_i B_k
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        # r=0: A0 B0 -> C00
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1 # k=0, i=0 -> 0*2+0=0
        
        # r=1: A0 B1 -> C01
        U[0, 1] = 1
        V[1, 1] = 1
        W[1, 1] = 1 # k=1, i=0 -> 1*2+0=2? No.
        # W index is k*n+i. n=2.
        # C_ik. i is row, k is col.
        # W[k*n+i].
        # C_00: k=0, i=0 -> 0.
        # C_01: k=1, i=0 -> 1*2+0=2.
        # C_10: k=0, i=1 -> 0*2+1=1.
        # C_11: k=1, i=1 -> 1*2+1=3.
        
        W[2, 1] = 1
        
        # r=2: A1 B0 -> C10
        U[1, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # r=3: A1 B1 -> C11
        U[1, 3] = 1
        V[1, 3] = 1
        W[3, 3] = 1
        
        return U, V, W

    def block_232():
        # 2x3x2
        # Split A (2x3) into A1 (2x1) and A2 (2x2)
        # Split B (3x2) into B1 (1x2) and B2 (2x2)
        # C = A1B1 + A2B2
        
        U1, V1, W1 = block_212()
        U2, V2, W2 = strassen_222()
        
        # U1 is 2x4. A1 is cols 0.
        # U2 is 4x7. A2 is cols 1,2.
        # Local U for 2x3x2 is 6x11.
        U = np.zeros((6, 11))
        V = np.zeros((6, 11))
        W = np.zeros((4, 11))
        
        # Place U1
        U[0:2, 0:4] = U1
        
        # Place U2 (A2 is cols 1,2 of A. So rows 2,3 and 4,5)
        U[2:4, 4:11] = U2[0:2, :]
        U[4:6, 4:11] = U2[2:4, :]
        
        # Place V1 (B1 is row 0 of B. So rows 0,1)
        V[0:2, 0:4] = V1
        
        # Place V2 (B2 is rows 1,2 of B. So rows 2,3 and 4,5)
        V[2:4, 4:11] = V2[0:2, :]
        V[4:6, 4:11] = V2[2:4, :]
        
        # W is just concatenation
        W[:, 0:4] = W1
        W[:, 4:11] = W2
        
        return U, V, W

    # Global assembly
    rank = 0
    U_global = np.zeros((n*m, 100)) # Pre-allocate large enough
    V_global = np.zeros((m*p, 100))
    W_global = np.zeros((p*n, 100))
    
    # Block 1: 2x2x2 at A[0:2, 0:2], B[0:2, 0:2] -> C[0:2, 0:2]
    u, v, w = strassen_222()
    # U rows: 0,1,2,3 (A[0,0], A[0,1], A[1,0], A[1,1])
    U_global[0:4, rank:rank+7] = u
    # V rows: 0,1,2,3 (B[0,0], B[0,1], B[1,0], B[1,1])
    V_global[0:4, rank:rank+7] = v
    # W rows: 0,1,2,3 (C[0,0], C[0,1], C[1,0], C[1,1])
    W_global[0:4, rank:rank+7] = w
    rank += 7
    
    # Block 2: 2x2x2 at A[0:2, 0:2], B[0:2, 2:4] -> C[0:2, 2:4]
    u, v, w = strassen_222()
    U_global[0:4, rank:rank+7] = u
    # V rows: B[0,2], B[0,3], B[1,2], B[1,3] -> indices 2,3,4,5
    V_global[2:6, rank:rank+7] = v
    # W rows: C[0,2], C[0,3], C[1,2], C[1,3] -> k=2,3; i=0,1
    # k=2, i=0 -> 4. k=2, i=1 -> 5. k=3, i=0 -> 6. k=3, i=1 -> 7.
    W_global[4:8, rank:rank+7] = w
    rank += 7
    
    # Block 3: 2x2x2 at A[0:2, 0:2], B[0:2, 4:6] -> C[0:2, 4:6]
    u, v, w = strassen_222()
    U_global[0:4, rank:rank+7] = u
    # V rows: B[0,4], B[0,5], B[1,4], B[1,5] -> indices 4,5,6,7
    V_global[4:8, rank:rank+7] = v
    # W rows: C[0,4], C[0,5], C[1,4], C[1,5] -> k=4,5; i=0,1
    # k=4, i=0 -> 8. k=4, i=1 -> 9. k=5, i=0 -> 10. k=5, i=1 -> 11.
    W_global[8:12, rank:rank+7] = w
    rank += 7
    
    # Block 4: 2x3x2 at A[0:2, 2:5], B[2:5, 0:2] -> C[0:2, 0:2]
    u, v, w = block_232()
    # U rows: A[0,2], A[0,3], A[0,4], A[1,2], A[1,3], A[1,4]
    # Indices: 2,3,4, 7,8,9
    U_global[2:5, rank:rank+11] = u[0:3, :]
    U_global[7:10, rank:rank+11] = u[3:6, :]
    
    # V rows: B[2,0], B[2,1], B[3,0], B[3,1], B[4,0], B[4,1]
    # Indices: 10,11, 12,13, 14,15
    V_global[10:13, rank:rank+11] = v[0:3, :]
    V_global[13:16, rank:rank+11] = v[3:6, :]
    
    # W rows: C[0,0], C[0,1], C[1,0], C[1,1] -> 0,1,2,3
    W_global[0:4, rank:rank+11] = w
    rank += 11
    
    # Block 5: 2x3x2 at A[0:2, 2:5], B[2:5, 2:4] -> C[0:2, 2:4]
    u, v, w = block_232()
    U_global[2:5, rank:rank+11] = u[0:3, :]
    U_global[7:10, rank:rank+11] = u[3:6, :]
    
    # V rows: B[2,2], B[2,3], B[3,2], B[3,3], B[4,2], B[4,3]
    # Indices: 12,13, 14,15, 16,17
    V_global[12:15, rank:rank+11] = v[0:3, :]
    V_global[15:18, rank:rank+11] = v[3:6, :]
    
    # W rows: C[0,2], C[0,3], C[1,2], C[1,3] -> 4,5,6,7
    W_global[4:8, rank:rank+11] = w
    rank += 11
    
    # Block 6: 2x3x2 at A[0:2, 2:5], B[2:5, 4:6] -> C[0:2, 4:6]
    u, v, w = block_232()
    U_global[2:5, rank:rank+11] = u[0:3, :]
    U_global[7:10, rank:rank+11] = u[3:6, :]
    
    # V rows: B[2,4], B[2,5], B[3,4], B[3,5], B[4,4], B[4,5]
    # Indices: 14,15, 16,17, 18,19
    V_global[14:17, rank:rank+11] = v[0:3, :]
    V_global[17:20, rank:rank+11] = v[3:6, :]
    
    # W rows: C[0,4], C[0,5], C[1,4], C[1,5] -> 8,9,10,11
    W_global[8:12, rank:rank+11] = w
    rank += 11
    
    return {
        "U": U_global[:, :rank].tolist(),
        "V": V_global[:, :rank].tolist(),
        "W": W_global[:, :rank].tolist()
    }

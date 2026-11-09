"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Helper to combine decompositions
    def combine(U1, V1, W1, U2, V2, W2):
        # Concatenate along rank dimension
        U = np.hstack([U1, U2])
        V = np.hstack([V1, V2])
        W = np.hstack([W1, W2])
        return U, V, W

    def strassen_2x2x2():
        # Returns U, V, W for 2x2x2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1 # A11, A22
        V[0, 0] = 1; V[3, 0] = 1 # B11, B22
        W[0, 0] = 1; W[3, 0] = 1 # C11, C22 ? No, W maps to output.
        
        # The condition is sum_r U_ij_r V_jk_r W_ki_r = 1.
        # This implies W_ki_r must be 1 for the specific i,k?
        # No, W is part of the tensor.
        # For Strassen, the tensor decomposition is known.
        # I will use the standard explicit factors.
        
        # Reference:
        # U (A factors):
        # r=1: A11+A22 -> U[0,0]=1, U[3,0]=1
        # r=2: A21+A22 -> U[2,1]=1, U[3,1]=1
        # r=3: A11 -> U[0,2]=1
        # r=4: A22 -> U[3,3]=1
        # r=5: A11+A12 -> U[0,4]=1, U[1,4]=1
        # r=6: A21-A11 -> U[2,5]=1, U[0,5]=-1
        # r=7: A12-A22 -> U[1,6]=1, U[3,6]=-1
        
        U = np.array([
            [1, 0, 1, 0, 1, -1, 0], # i=0,j=0 (A11)
            [0, 0, 0, 0, 1, 0, 1],  # i=0,j=1 (A12)
            [0, 1, 0, 0, 0, 1, 0],  # i=1,j=0 (A21)
            [1, 1, 0, 1, 0, 0, -1]  # i=1,j=1 (A22)
        ])
        
        # V (B factors):
        # r=1: B11+B22 -> V[0,0]=1, V[3,0]=1
        # r=2: B11 -> V[0,1]=1
        # r=3: B12-B22 -> V[1,2]=1, V[3,2]=-1
        # r=4: B21-B11 -> V[2,3]=1, V[0,3]=-1
        # r=5: B22 -> V[3,4]=1
        # r=6: B11+B12 -> V[0,5]=1, V[1,5]=1
        # r=7: B21+B22 -> V[2,6]=1, V[3,6]=1
        
        V = np.array([
            [1, 1, 0, -1, 0, 1, 0], # j=0,k=0 (B11)
            [0, 0, 1, 0, 0, 1, 0],  # j=0,k=1 (B12)
            [0, 0, 0, 1, 0, 0, 1],  # j=1,k=0 (B21)
            [1, 0, -1, 0, 1, 0, 1]  # j=1,k=1 (B22)
        ])
        
        # W (C factors):
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # W[k*2+i, r]
        # k=0,i=0 (C11): W[0, r]
        # k=0,i=1 (C12): W[1, r]
        # k=1,i=0 (C21): W[2, r]
        # k=1,i=1 (C22): W[3, r]
        
        W = np.array([
            [1, 0, 0, 1, -1, 0, 1], # C11
            [0, 0, 1, 0, 1, 0, 0],  # C12
            [0, 1, 0, 1, 0, 0, 0],  # C21
            [1, -1, 1, 0, 0, 1, 0]  # C22
        ])
        
        return U, V, W

    # Base: 2x2x2
    U2, V2, W2 = strassen_2x2x2()
    
    # Construct 3x2x2
    # A (3x2), B (2x2)
    # Split A into A_top (2x2) and A_bot (1x2)
    # C = A_top B + A_bot B
    # Rank = R(2,2,2) + R(1,2,2)
    
    # R(1,2,2) trivial:
    # A (1x2), B (2x2)
    # C11 = A1 B11 + A2 B21
    # C12 = A1 B12 + A2 B22
    # U (2x2):
    # r1: A1 -> U[0,0]=1
    # r2: A2 -> U[1,1]=1
    # V (2x2):
    # r1: B11, B21? No.
    # C11 uses B11, B21. C12 uses B12, B22.
    # r1: B11, B21? No, V is indexed by j,k.
    # j=0,k=0 (B11), j=1,k=0 (B21).
    # For C11 (k=0,i=0): W[0,0]=1.
    # V[0,0]=1 (B11), V[1,0]=1 (B21).
    # U[0,0]=1 (A1).
    # Term: 1*1*1 = 1.
    # But we need A1 B11 + A2 B21.
    # So we need two terms for C11?
    # r1: A1 B11. U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # r2: A2 B21. U[1,1]=1, V[1,0]=1, W[0,1]=1.
    # r3: A1 B12. U[0,2]=1, V[0,1]=1, W[1,2]=1.
    # r4: A2 B22. U[1,3]=1, V[1,1]=1, W[1,3]=1.
    
    U_122 = np.zeros((2, 4))
    V_122 = np.zeros((4, 4))
    W_122 = np.zeros((4, 4))
    
    U_122[0, 0] = 1
    V_122[0, 0] = 1
    W_122[0, 0] = 1
    
    U_122[1, 1] = 1
    V_122[1, 0] = 1
    W_122[0, 1] = 1
    
    U_122[0, 2] = 1
    V_122[0, 1] = 1
    W_122[1, 2] = 1
    
    U_122[1, 3] = 1
    V_122[1, 1] = 1
    W_122[1, 3] = 1
    
    # Combine U2 (4x7) and U_122 (2x4) for 3x2x2?
    # No, U2 is for 2x2x2 (A is 2x2).
    # We need to embed U2 into 3x2x2.
    # A_top (2x2) uses U2. A_bot (1x2) uses U_122.
    # U_322 shape (6, 11).
    U_322 = np.zeros((6, 11))
    V_322 = np.zeros((4, 11))
    W_322 = np.zeros((4, 11))
    
    # First 7 columns: Strassen
    U_322[0:4, 0:7] = U2
    V_322[:, 0:7] = V2
    W_322[:, 0:7] = W2
    
    # Last 4 columns: 1x2x2
    U_322[4:6, 7:11] = U_122
    V_322[:, 7:11] = V_122
    W_322[:, 7:11] = W_122
    
    # Construct 3x4x2
    # A (3x4), B (4x2)
    # Split A into A_L (3x2), A_R (3x2)
    # Split B into B_L (2x2), B_R (2x2)
    # C = A_L B_L + A_R B_R
    # Rank = 2 * R(3,2,2) = 18
    
    U_342 = np.zeros((12, 18))
    V_342 = np.zeros((8, 18))
    W_342 = np.zeros((4, 18))
    
    # First 9: A_L B_L
    U_342[0:6, 0:9] = U_322
    V_342[0:4, 0:9] = V_322
    W_342[:, 0:9] = W_322
    
    # Last 9: A_R B_R
    U_342[6:12, 9:18] = U_322
    V_342[4:8, 9:18] = V_322
    W_342[:, 9:18] = W_322
    
    # Construct 3x4x4
    # A (3x4), B (4x4)
    # Split B into B_L (4x2), B_R (4x2)
    # C_L = A B_L, C_R = A B_R
    # Rank = 2 * R(3,4,2) = 36
    
    U_344 = np.zeros((12, 36))
    V_344 = np.zeros((16, 36))
    W_344 = np.zeros((8, 36))
    
    U_344[:, 0:18] = U_342
    V_344[0:8, 0:18] = V_342
    W_344[0:4, 0:18] = W_342
    
    U_344[:, 18:36] = U_342
    V_344[8:16, 18:36] = V_342
    W_344[4:8, 18:36] = W_342
    
    # Construct 3x4x8
    # A (3x4), B (4x8)
    # Split B into B_L (4x4), B_R (4x4)
    # C_L = A B_L, C_R = A B_R
    # Rank = 2 * R(3,4,4) = 72
    
    U_final = np.zeros((12, 72))
    V_final = np.zeros((32, 72))
    W_final = np.zeros((24, 72))
    
    U_final[:, 0:36] = U_344
    V_final[0:16, 0:36] = V_344
    W_final[0:8, 0:36] = W_344
    
    U_final[:, 36:72] = U_344
    V_final[16:32, 36:72] = V_344
    W_final[8:24, 36:72] = W_344
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END

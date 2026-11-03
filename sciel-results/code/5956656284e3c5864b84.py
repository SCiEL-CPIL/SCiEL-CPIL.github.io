import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen factors for 2x2x2
    # U, V, W are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C10
    W2[3, 1] = -1 # C11
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C01
    W2[3, 2] = 1 # C11
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1 # C00
    W2[2, 3] = 1 # C10
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C00
    W2[1, 4] = 1 # C01
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00
    
    # 2. Construct 4x4x4 factors (Rank 49)
    # U4 is 16x49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U4[i, r] = U2[bi, r1] * U2[ii, r2]
            for bi in range(2):
                for ii in range(2):
                    idx_a = bi * 2 + ii
                    U4[idx_a, r] = U2[bi, r1] * U2[ii, r2]
                    
            for bj in range(2):
                for jj in range(2):
                    idx_b = bj * 2 + jj
                    V4[idx_b, r] = V2[bj, r1] * V2[jj, r2]
                    
            for ck in range(2):
                for kk in range(2):
                    idx_c = ck * 2 + kk
                    W4[idx_c, r] = W2[ck, r1] * W2[kk, r2]
                    
    # 3. Schoolbook 4x4x1 factors (Rank 4)
    U_sb = np.zeros((16, 4))
    V_sb = np.zeros((4, 4))
    W_sb = np.zeros((4, 4))
    
    for i in range(4):
        for j in range(4):
            U_sb[i * 4 + j, j] = 1
            V_sb[j, j] = 1
            W_sb[j, j] = 1 # W[k, r] where k=i? No.
            
    # Wait, W_sb index is k. C is 4x1. k is row index of C.
    # C_k = sum_j A_kj B_j.
    # W[k, r] = 1 if k=r.
    # So W_sb is Identity.
    
    # 4. Assemble 4x4x9 factors (Rank 102)
    rank = 102
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # C1 (cols 0-48)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # C2 (cols 49-97)
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    # C3 (cols 98-101)
    U[:, 98:102] = U_sb
    V[32:36, 98:102] = V_sb
    W[32:36, 98:102] = W_sb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

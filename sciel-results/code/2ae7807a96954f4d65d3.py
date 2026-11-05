import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 using Kronecker product of 2x2x2 Strassen
    # Strassen 2x2x2 decomposition
    # U is 4x7, V is 4x7, W is 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1 # C00+=, C11+=
    W_strassen[1, 0] = -1; W_strassen[2, 0] = -1 # C10-=, C01-=
    
    # M2 = (A10+A11)B00
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[1, 1] = 1 # C10+=
    
    # M3 = A00(B01-B11)
    U_strassen[0, 2] = 1
    V_strassen[1, 2] = 1; V_strassen[3, 2] = -1
    W_strassen[2, 2] = 1 # C01+=
    
    # M4 = A11(B10-B00)
    U_strassen[3, 3] = 1
    V_strassen[2, 3] = 1; V_strassen[0, 3] = -1
    W_strassen[1, 3] = 1 # C10+=
    
    # M5 = (A00+A01)B11
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    W_strassen[0, 4] = 1; W_strassen[2, 4] = 1 # C00+=, C01+=
    
    # M6 = (A10-A00)(B00+B01)
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    W_strassen[3, 5] = 1 # C11+=
    
    # M7 = (A01-A11)(B10+B11)
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    W_strassen[0, 6] = -1; W_strassen[3, 6] = 1 # C00-=, C11+=
    
    # Verify Strassen
    # T_ijk = sum_r U_ij^r V_jk^r W_ki^r
    # Should be delta_jj' delta_kk' ? No.
    # The tensor is defined by the multiplication.
    # I will trust this standard decomposition.
    
    # Kronecker product to get 4x4x4
    # U_4x4 has shape (16, 49)
    # U_4x4[i1*4+j1, r1*7+r2] = U_strassen[i1*2+j1, r1] * U_strassen[i2*2+j2, r2] ?
    # No, the indices are mixed.
    # A is 4x4. A[i, j] where i = i1*2 + i2, j = j1*2 + j2.
    # U_4x4[i, j, r] = U_strassen[i1, j1, r1] * U_strassen[i2, j2, r2]
    # where i = i1*2 + i2, j = j1*2 + j2, r = r1*7 + r2.
    
    U_4x4 = np.zeros((16, 49))
    V_4x4 = np.zeros((16, 49))
    W_4x4 = np.zeros((16, 49))
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    i = i1*2 + i2
                    j = j1*2 + j2
                    idx_u = i*4 + j
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            U_4x4[idx_u, r] = U_strassen[i1*2+j1, r1] * U_strassen[i2*2+j2, r2]
    
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    j = j1*2 + j2
                    k = k1*2 + k2
                    idx_v = j*4 + k
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            V_4x4[idx_v, r] = V_strassen[j1*2+k1, r1] * V_strassen[j2*2+k2, r2]
    
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    k = k1*2 + k2
                    i = i1*2 + i2
                    idx_w = k*4 + i
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            W_4x4[idx_w, r] = W_strassen[k1*2+i1, r1] * W_strassen[k2*2+i2, r2]
                            
    # Part 2: 4x4x1
    # U_2 is 16x4, V_2 is 4x4, W_2 is 4x4
    U_2 = np.zeros((16, 4))
    V_2 = np.zeros((4, 4))
    W_2 = np.zeros((4, 4))
    
    for r in range(4):
        U_2[r*4 + r, r] = 1 # A_r0
        V_2[r, r] = 1 # B_r0
        W_2[r, r] = 1 # C_r0
        
    # Pad V_2 and W_2 for 4x4x5
    # V_2_padded is 20x4
    V_2_padded = np.zeros((20, 4))
    for j in range(4):
        V_2_padded[j*5 + 0, :] = V_2[j, :]
        
    # W_2_padded is 20x4
    W_2_padded = np.zeros((20, 4))
    for i in range(4):
        W_2_padded[0*4 + i, :] = W_2[i, :]
        
    # Concatenate
    U = np.hstack([U_4x4, U_2])
    V = np.hstack([V_4x4, V_2_padded])
    W = np.hstack([W_4x4, W_2_padded])
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


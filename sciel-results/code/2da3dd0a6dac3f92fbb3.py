import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices
    # U, V, W are 2x2x7 tensors (flattened to 2D matrices)
    # U[i, j, r], V[j, k, r], W[k, i, r]
    
    # Define Strassen matrices
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices 0, 1 for 2x2
    # U[i, j, r]
    U2 = np.zeros((2, 2, 7), dtype=int)
    V2 = np.zeros((2, 2, 7), dtype=int)
    W2 = np.zeros((2, 2, 7), dtype=int) # W[k, i, r]
    
    # r=0 (M1)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    
    # r=1 (M2)
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1 # C[1,0] gets M2? C[1,0] = M2+M4. Yes.
    
    # r=2 (M3)
    U2[0, 0, 2] = 1
    V2[1, 0, 2] = 1; V2[1, 1, 2] = -1 # B21 - B22? B[j, k]. j=1, k=0 is B11? No.
    # B is 2x2. B[j, k].
    # M3 = A11 (B21 - B22).
    # B21 is B[1, 0]. B22 is B[1, 1].
    # So V[1, 0, 2] = 1, V[1, 1, 2] = -1.
    W2[0, 0, 2] = 1 # C[0,0] gets M3? C[0,0] = M1+M4-M5+M7. No.
    # C[0,1] = M3+M5.
    # C[1,1] = M1-M2+M3+M6.
    # So M3 contributes to C[0,1] and C[1,1].
    # C[0,1] -> i=0, k=1. W[1, 0, 2] = 1.
    # C[1,1] -> i=1, k=1. W[1, 1, 2] = 1.
    W2[1, 0, 2] = 1
    W2[1, 1, 2] = 1
    
    # r=3 (M4)
    U2[1, 1, 3] = 1
    V2[0, 1, 3] = 1; V2[0, 0, 3] = -1 # B12 - B11. B[0,1] - B[0,0].
    W2[0, 0, 3] = 1 # C[0,0] gets M4.
    W2[0, 1, 3] = 1 # C[1,0] gets M4.
    
    # r=4 (M5)
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1 # B22. B[1,1].
    W2[0, 0, 4] = -1 # C[0,0] gets -M5.
    W2[0, 1, 4] = 1  # C[0,1] gets M5.
    
    # r=5 (M6)
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1 # A21 - A11.
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1 # B11 + B12.
    W2[1, 1, 5] = 1 # C[1,1] gets M6.
    
    # r=6 (M7)
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1 # A12 - A22.
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1 # B21 + B22.
    W2[0, 0, 6] = 1 # C[0,0] gets M7.
    
    # Flatten to matrices
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    U2_mat = U2.reshape(4, 7)
    V2_mat = V2.reshape(4, 7)
    W2_mat = W2.reshape(4, 7)
    
    # Construct 4x4 decomposition
    # Rank 49
    R = 49
    U4 = np.zeros((16, R), dtype=int)
    V4 = np.zeros((16, R), dtype=int)
    W4 = np.zeros((16, R), dtype=int)
    
    for r1 in range(7):
        for r2 in range(7):
            r4 = r1 * 7 + r2
            
            # U4 indices: i4, j4. i4 = i1*2 + i2, j4 = j1*2 + j2
            # U4[i4, j4, r4] = U2[i1, j1, r1] * U2[i2, j2, r2]
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2_mat[i1*2 + j1, r1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2_mat[i2*2 + j2, r2]
                            if u2 == 0: continue
                            i4 = i1*2 + i2
                            j4 = j1*2 + j2
                            idx_u = i4*4 + j4
                            U4[idx_u, r4] += u1 * u2
            
            # V4 indices: j4, k4. j4 = j1*2 + j2, k4 = k1*2 + k2
            # V4[j4, k4, r4] = V2[j1, k1, r1] * V2[j2, k2, r2]
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V2_mat[j1*2 + k1, r1]
                    if v1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2 = V2_mat[j2*2 + k2, r2]
                            if v2 == 0: continue
                            j4 = j1*2 + j2
                            k4 = k1*2 + k2
                            idx_v = j4*4 + k4
                            V4[idx_v, r4] += v1 * v2
            
            # W4 indices: k4, i4. k4 = k1*2 + k2, i4 = i1*2 + i2
            # W4[k4, i4, r4] = W2[k1, i1, r1] * W2[k2, i2, r2]
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W2_mat[k1*2 + i1, r1]
                    if w1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w2 = W2_mat[k2*2 + i2, r2]
                            if w2 == 0: continue
                            k4 = k1*2 + k2
                            i4 = i1*2 + i2
                            idx_w = k4*4 + i4
                            W4[idx_w, r4] += w1 * w2

    # Convert to lists for the required interface
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

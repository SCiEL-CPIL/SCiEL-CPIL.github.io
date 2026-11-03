import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook for 4x4x4 (rank 64) + Rank 4 for 4x4x1
    # Total rank 68
    
    # Primary: Strassen 2x2x2 (rank 7) -> 4x4x4 (rank 49) + Rank 4 for 4x4x1
    # Total rank 53
    
    # Define Strassen 2x2x2
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # c=0: (a+d)(e+h)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # c=1: (c+d)e
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # c=2: a(f-h)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # c=3: d(g-c)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # c=4: (a+b)h
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # c=5: (c-d)(f+g)
    U2[2, 5] = 1; U2[3, 5] = -1
    V2[1, 5] = 1; V2[2, 5] = 1
    W2[3, 5] = 1
    
    # c=6: (b-d)(e+g)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[0, 6] = 1; V2[2, 6] = 1
    W2[0, 6] = 1

    # Kronecker product for 4x4x4
    # U4[i*2+j, c*7+d] = U2[i,j,c] * U2[j,k,d]? 
    # No, Kronecker product of tensors.
    # T4[i1 i2, j1 j2, k1 k2] = T2[i1, j1, k1] * T2[i2, j2, k2]
    # U4[(i1,i2), (j1,j2), (c1,c2)] = U2[i1, j1, c1] * U2[i2, j2, c2]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            val_u = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
                            if val_u != 0:
                                U4[i1*4+i2, j1*4+j2, c] = val_u
                                
                            val_v = V2[j1*2+k1, c1] * V2[j2*2+k2, c2] # Wait, k is not in loop
                            # V2 is indexed by (j,k).
                            # We need to iterate k1, k2 as well for V and W.
                            
    # Better way to construct Kronecker:
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    u_val = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
                                    v_val = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
                                    w_val = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
                                    
                                    if u_val != 0 and v_val != 0 and w_val != 0:
                                        i = i1*2 + i2
                                        j = j1*2 + j2
                                        k = k1*2 + k2
                                        U4[i*4+j, c] = u_val
                                        V4[j*4+k, c] = v_val
                                        W4[k*4+i, c] = w_val
                                        
    # Construct 4x4x1 part (rank 4)
    U1 = np.zeros((16, 4))
    V1 = np.zeros((20, 4))
    W1 = np.zeros((20, 4))
    
    for c in range(4):
        for i in range(4):
            for j in range(4):
                # U[i, j, c] = delta(i, c)
                if i == c:
                    U1[i*4+j, c] = 1
                # V[j, 4, c] = 1 (k=4 is the 5th column, index 4)
                V1[j*5+4, c] = 1
                # W[4, i, c] = 1
                W1[4*4+i, c] = 1
                
    # Combine U4, V4, W4 with U1, V1, W1
    # U4 is 16x49, U1 is 16x4
    U = np.hstack([U4, U1])
    # V4 is 16x49, V1 is 20x4
    # V4 corresponds to k=0..3. V1 corresponds to k=4.
    # V4 has shape (16, 49). Rows are j*4+k.
    # V1 has shape (20, 4). Rows are j*5+k.
    # We need to align them.
    # V4 rows: j in 0..3, k in 0..3. Index j*4+k.
    # V1 rows: j in 0..3, k=4. Index j*5+4.
    # We need to map V4 to a 20-row matrix.
    V_full = np.zeros((20, 49))
    for j in range(4):
        for k in range(4):
            V_full[j*5+k, :] = V4[j*4+k, :]
    V = np.hstack([V_full, V1])
    
    # W4 is 16x49. Rows are k*4+i.
    # W1 is 20x4. Rows are k*4+i.
    # W4 rows: k in 0..3, i in 0..3.
    # W1 rows: k=4, i in 0..3.
    W_full = np.zeros((20, 49))
    for k in range(4):
        for i in range(4):
            W_full[k*4+i, :] = W4[k*4+i, :]
    W = np.hstack([W_full, W1])
    
    # Verify
    def is_valid(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 0
                    for c in range(U.shape[1]):
                        lhs += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(lhs - 1.0) > 1e-9:
                        return False
        return True
        
    if is_valid(U.tolist(), V.tolist(), W.tolist()):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to Rank 68
        U_fb = np.zeros((16, 68))
        V_fb = np.zeros((20, 68))
        W_fb = np.zeros((20, 68))
        
        # Schoolbook 4x4x4 (rank 64)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    c = i*16 + j*4 + k
                    U_fb[i*4+j, c] = 1
                    V_fb[j*5+k, c] = 1
                    W_fb[k*4+i, c] = 1
                    
        # Rank 4 4x4x1
        for c in range(4):
            c_idx = 64 + c
            for i in range(4):
                for j in range(4):
                    if i == c:
                        U_fb[i*4+j, c_idx] = 1
                    V_fb[j*5+4, c_idx] = 1
                    W_fb[4*4+i, c_idx] = 1
                    
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END

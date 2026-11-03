import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4x4 using Strassen (Kronecker product of 2x2x2 Strassen)
    # Strassen 2x2x2 matrices
    # U2, V2, W2 are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ])
    W2 = np.array([
        [1, 0, 0, 0, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, 0, 0, 0, 0, 1, 1]   # C11
    ])
    
    rank_44 = 49
    U44 = np.zeros((16, rank_44))
    V44 = np.zeros((16, rank_44))
    W44 = np.zeros((16, rank_44))
    
    # Kronecker product
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    u_idx = i1 * 2 + i2
                    v_idx = j1 * 2 + j2
                    for r1 in range(7):
                        for r2 in range(7):
                            r_idx = r1 * 7 + r2
                            U44[u_idx, r_idx] = U2[i1, j1, r1] * U2[i2, j2, r2]
                            V44[v_idx, r_idx] = V2[i1, j1, r1] * V2[i2, j2, r2]
                            W44[v_idx, r_idx] = W2[i1, j1, r1] * W2[i2, j2, r2] # Note: W index is (k,i). 
                            # In 2x2x2, W2 is indexed by (k,i). 
                            # In 4x4x4, W index is (k1*2+k2, i1*2+i2).
                            # The Kronecker product structure for W is W[k1,k2, i1,i2, r1,r2] = W2[k1, i1, r1] * W2[k2, i2, r2].
                            # My loop above used v_idx for W44 row, which is j. This is wrong.
                            # W44 row index should be k.
                            pass
                            
    # Let's redo the Kronecker construction correctly.
    # U44[i1,i2, j1,j2, r1,r2] = U2[i1, j1, r1] * U2[i2, j2, r2]
    # V44[i1,i2, j1,j2, r1,r2] = V2[i1, j1, r1] * V2[i2, j2, r2]  (Note: V input is B, so indices are j,k? No, V is for B. B is m x p. So V index is (j, k). In 2x2x2, V2 is 4x7, indexed by (j, k) where j,k in 0,1.
    # In 4x4x4, B is 4x4. j in 0..3, k in 0..3.
    # V44[j1,j2, k1,k2, r1,r2] = V2[j1, k1, r1] * V2[j2, k2, r2]
    # W44[k1,k2, i1,i2, r1,r2] = W2[k1, i1, r1] * W2[k2, i2, r2]
    
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    u_row = i1 * 2 + i2
                    v_row = j1 * 2 + j2
                    for r1 in range(7):
                        for r2 in range(7):
                            r_col = r1 * 7 + r2
                            U44[u_row, r_col] = U2[i1, j1, r1] * U2[i2, j2, r2]
                            V44[v_row, r_col] = V2[j1, i1, r1] * V2[j2, i2, r2] # Wait, V2 indices?
                            # In 2x2x2, V2 is for B. B is 2x2.
                            # V2[j, k, r].
                            # So V44[j1,j2, k1,k2, r1,r2] = V2[j1, k1, r1] * V2[j2, k2, r2].
                            # But my loop iterates j1, j2. Where is k?
                            # The rank decomposition is a sum of rank-1 tensors.
                            # The term r corresponds to a specific r1, r2.
                            # The U part picks A[i, j]. The V part picks B[j, k]. The W part picks C[k, i].
                            # So for a fixed r (r1, r2):
                            # U[i, j, r] = U2[i1, j1, r1] * U2[i2, j2, r2]
                            # V[j, k, r] = V2[j1, k1, r1] * V2[j2, k2, r2]
                            # W[k, i, r] = W2[k1, i1, r1] * W2[k2, i2, r2]
                            
                            # My loop structure needs to iterate over all i,j,k to fill U,V,W.
                            # But U depends only on i,j. V on j,k. W on k,i.
                            # So I can fill them in separate loops or one big loop.
                            pass
                            
    # Correct construction:
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            U44[i, r] = U2[i1, j1, r1] * U2[i2, j2, r2]
                            
                            # V44 needs k. 
                            # V2[j, k, r]. 
                            # So for each j, we need to fill V44[j, r] for all k?
                            # No, V44 is 16x49. Rows are j (0..3)? No, V is for B (m x p). 
                            # m=4, p=4. So V is 16x49.
                            # Row index is j*4+k? Or j+p*k?
                            # Interface: V is (m*p, rank). 
                            # Usually V[j*p + k, r] or V[k*m + j, r]?
                            # Task says: V of shape (m*p, rank). 
                            # Standard is V[j, k] -> index j*p + k? Or k*m + j?
                            # Let's check U: (n*m, rank). U[i, j] -> i*m + j.
                            # W: (p*n, rank). W[k, i] -> k*n + i.
                            # So V is likely V[j, k] -> j*p + k.
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    v_row = j * 4 + k
                                    V44[v_row, r] = V2[j1, k1, r1] * V2[j2, k2, r2]
                                    
                                    w_row = k * 4 + i
                                    W44[w_row, r] = W2[k1, i1, r1] * W2[k2, i2, r2]
                                    
    # Part 2: 4x4x3 using Schoolbook
    rank_443 = 48
    U443 = np.zeros((16, rank_443))
    V443 = np.zeros((16, rank_443))
    W443 = np.zeros((12, rank_443)) # p=3, n=4 -> 12
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U443[i * 4 + j, col] = 1
                V443[j * 3 + k, col] = 1
                W443[k * 4 + i, col] = 1
                col += 1
                
    # Combine
    total_rank = 49 + 48
    U_total = np.hstack([U44, U443])
    V_total = np.hstack([V44, V443])
    
    W_total = np.zeros((28, total_rank))
    # W44 corresponds to k=0..3. Indices 0..15.
    W_total[0:16, 0:49] = W44
    # W443 corresponds to k=4..6. Indices 16..27.
    # W443 row index is k_local * 4 + i, where k_local in 0..2.
    # Global k = 4 + k_local.
    # Global index = (4 + k_local) * 4 + i = 16 + k_local * 4 + i.
    W_total[16:28, 49:97] = W443
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

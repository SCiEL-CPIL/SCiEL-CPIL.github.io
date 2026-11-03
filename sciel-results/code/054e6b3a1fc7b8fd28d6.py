import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # 1. Construct 4x4x4 decomposition (Rank 49)
    # Kronecker product of two Strassen tensors
    # Index mapping: i = 2*i1 + i2, c = 7*c1 + c2
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    for i1 in range(2):
        for i2 in range(2):
            i = 2*i1 + i2
            for c1 in range(7):
                for c2 in range(7):
                    c = 7*c1 + c2
                    U_444[i, c] = U_s[i1, c1] * U_s[i2, c2]
                    V_444[i, c] = V_s[i1, c1] * V_s[i2, c2]
                    W_444[i, c] = W_s[i1, c1] * W_s[i2, c2]
    
    # 2. Construct 4x4x2 decomposition (Rank 28)
    # 4x4x2 is isomorphic to (2x2x1) x (2x2x2)
    # R(2,2,1) = 4. R(2,2,2) = 7. Total 28.
    # Or simply use 4 independent 2x2x2 multiplications? No, that's for 4x4x4.
    # For 4x4x2, we can use the Kronecker product of a rank-4 tensor for 2x2x1 and rank-7 for 2x2x2.
    
    # 2x2x1 tensor: A(2x2) * B(2x1) = C(2x1)
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    # Rank 4 decomposition:
    # T = sum_{r=1}^4 u_r v_r w_r
    # u: 4x4 (A indices), v: 2x4 (B indices), w: 2x4 (C indices)
    # Actually, we can just use the standard 4 multiplications.
    # U_221 (4x4), V_221 (2x4), W_221 (2x4)
    U_221 = np.eye(4)
    V_221 = np.eye(4)
    W_221 = np.eye(4)
    # Map indices:
    # A indices: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    # B indices: 0,1 -> (0,0), (1,0) ? No, B is 2x1. Indices 0,1.
    # C indices: 0,1 -> (0,0), (1,0).
    
    # Standard mult:
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    
    # Term 1: A_00 B_0 -> C_0
    U_221[0, 0] = 1; V_221[0, 0] = 1; W_221[0, 0] = 1
    # Term 2: A_01 B_1 -> C_0
    U_221[1, 1] = 1; V_221[1, 1] = 1; W_221[0, 1] = 1
    # Term 3: A_10 B_0 -> C_1
    U_221[2, 2] = 1; V_221[0, 2] = 1; W_221[1, 2] = 1
    # Term 4: A_11 B_1 -> C_1
    U_221[3, 3] = 1; V_221[1, 3] = 1; W_221[1, 3] = 1
    
    # Now Kronecker product of 2x2x1 (Rank 4) and 2x2x2 (Rank 7) to get 4x4x2 (Rank 28)
    # 4x4x2: A(4x4), B(4x2), C(4x2)
    # A is 2x2 blocks of 2x2.
    # B is 2x1 blocks of 2x2? No.
    # B is 4x2.
    # Let's map indices.
    # A: i = 2*i1 + i2, j = 2*j1 + j2.
    # B: j = 2*j1 + j2, k = 2*k1 + k2? No, B has 2 cols.
    # B indices: j (0..3), k (0..1).
    # C indices: i (0..3), k (0..1).
    
    # Tensor product structure:
    # (2x2x1) x (2x2x2)
    # 2x2x1: A1(2x2), B1(2x1), C1(2x1).
    # 2x2x2: A2(2x2), B2(2x2), C2(2x2).
    # Combined:
    # A = A1 x A2 (4x4).
    # B = B1 x B2 (4x2).
    # C = C1 x C2 (4x2).
    
    U_442 = np.zeros((16, 28))
    V_442 = np.zeros((8, 28))
    W_442 = np.zeros((8, 28))
    
    for c1 in range(4): # Rank of 2x2x1
        for c2 in range(7): # Rank of 2x2x2
            c = 7*c1 + c2
            # U: A indices
            for i1 in range(2):
                for i2 in range(2):
                    i = 2*i1 + i2
                    # U_221 index for A1: i1
                    # U_s index for A2: i2
                    U_442[i, c] = U_221[i1, c1] * U_s[i2, c2]
            
            # V: B indices
            # B1 is 2x1. Index j1.
            # B2 is 2x2. Indices j2, k2.
            # B combined: j = 2*j1 + j2? No.
            # B1 (2x1) and B2 (2x2).
            # B is 4x2.
            # B[j, k].
            # j = 2*j1 + j2.
            # k = k2.
            for j1 in range(2):
                for j2 in range(2):
                    for k2 in range(2):
                        j = 2*j1 + j2
                        k = k2
                        # V_221 index for B1: j1
                        # V_s index for B2: j2, k2?
                        # V_s is for 2x2x2. V_s[j, c].
                        # j in V_s is 0..3. j = 2*j2 + k2?
                        # No, V_s[j, c] where j is index in B (2x2).
                        # B2 is 2x2. Indices (j2, k2).
                        # V_s index = 2*j2 + k2.
                        v_s_idx = 2*j2 + k2
                        V_442[j*2+k, c] = V_221[j1, c1] * V_s[v_s_idx, c2]
                        
            # W: C indices
            # C1 is 2x1. Index i1.
            # C2 is 2x2. Indices i2, k2.
            # C combined: i = 2*i1 + i2, k = k2.
            for i1 in range(2):
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2*i1 + i2
                        k = k2
                        w_s_idx = 2*i2 + k2
                        W_442[k*4+i, c] = W_221[i1, c1] * W_s[w_s_idx, c2]
                        
    # Combine
    U_final = np.hstack([U_444, U_442])
    
    V_final = np.zeros((24, 77))
    V_final[0:16, 0:49] = V_444
    V_final[0:8, 49:77] = V_442
    
    W_final = np.zeros((24, 77))
    W_final[0:16, 0:49] = W_444
    W_final[0:8, 49:77] = W_442
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

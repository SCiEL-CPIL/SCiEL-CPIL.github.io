import numpy as np

def construct():
    n, m, p = 4, 5, 5
    # Rank 49 for M_4,4,4
    # Rank 16 for j=4, k=0..3
    # Rank 16 for M_4,4,1 (j=0..3, k=4)
    # Rank 4 for j=4, k=4
    
    rank_total = 49 + 16 + 16 + 4
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    # 1. Strassen for M_4,4,4 (j=0..3, k=0..3)
    # Define 2x2 Strassen vectors
    # u, v, w are 4x7
    u2 = np.zeros((4, 7))
    v2 = np.zeros((4, 7))
    w2 = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    u2[0, 0] = 1; u2[3, 0] = 1
    v2[0, 0] = 1; v2[3, 0] = 1
    w2[0, 0] = 1; w2[3, 0] = -1 # C11 = M1...
    # Actually, let's just use the standard Strassen decomposition.
    # C = [[C00, C01], [C10, C11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    # u2[i, k]
    
    # M1
    u2[0, 0] = 1; u2[3, 0] = 1
    v2[0, 0] = 1; v2[3, 0] = 1
    w2[0, 0] = 1; w2[3, 0] = 1 # C00 gets +1, C11 gets +1
    # Wait, w2 is for C. C is 2x2.
    # w2[0,0]=1 (C00), w2[3,0]=1 (C11).
    
    # M2
    u2[2, 1] = 1; u2[3, 1] = 1
    v2[0, 1] = 1
    w2[2, 1] = 1 # C10 gets +1
    
    # M3
    u2[0, 2] = 1
    v2[2, 2] = 1; v2[3, 2] = -1
    w2[1, 2] = 1 # C01 gets +1
    
    # M4
    u2[3, 3] = 1
    v2[1, 3] = 1; v2[0, 3] = -1
    w2[0, 3] = -1 # C00 gets -1
    w2[2, 3] = 1  # C10 gets +1
    
    # M5
    u2[0, 4] = 1; u2[1, 4] = 1
    v2[3, 4] = 1
    w2[0, 4] = 1 # C00 gets +1
    
    # M6
    u2[2, 5] = 1; u2[0, 5] = -1
    v2[0, 5] = 1; v2[1, 5] = 1
    w2[3, 5] = 1 # C11 gets +1
    
    # M7
    u2[1, 6] = 1; u2[3, 6] = -1
    v2[2, 6] = 1; v2[3, 6] = 1
    w2[0, 6] = -1 # C00 gets -1
    w2[1, 6] = 1  # C01 gets +1
    
    # Construct 4x4 vectors
    # u4 is 16x49
    u4 = np.zeros((16, 49))
    v4 = np.zeros((16, 49))
    w4 = np.zeros((16, 49))
    
    col = 0
    for i in range(7):
        for j in range(7):
            # Tensor product u2[:, i] and u2[:, j]
            # u2 is 4x7.
            # Result is 16x1.
            u4[:, col] = np.kron(u2[:, i], u2[:, j])
            v4[:, col] = np.kron(v2[:, i], v2[:, j])
            w4[:, col] = np.kron(w2[:, i], w2[:, j])
            col += 1
            
    # Map to U, V, W for 4x5x5
    # A is 4x5. u4 corresponds to 4x4 part (cols 0-3).
    # U index: i*5 + j.
    for r in range(49):
        for idx, val in enumerate(u4[:, r]):
            if val != 0:
                # idx is i*4 + j for 4x4
                i = idx // 4
                j = idx % 4
                U[i*5 + j, r] = val
                
        for idx, val in enumerate(v4[:, r]):
            if val != 0:
                # idx is j*4 + k for 4x4
                j = idx // 4
                k = idx % 4
                V[j*5 + k, r] = val
                
        for idx, val in enumerate(w4[:, r]):
            if val != 0:
                # idx is k*4 + i for 4x4
                k = idx // 4
                i = idx % 4
                W[k*4 + i, r] = val

    # 2. j=4, k=0..3 (16 terms)
    # C_{i,k} = A_{i,4} B_{4,k}
    # u: i*5+4 = 1
    # v: 4*5+k = 20+k
    # w: k*4+i
    col_offset = 49
    for i in range(4):
        for k in range(4):
            r = col_offset + i*4 + k
            U[i*5 + 4, r] = 1
            V[4*5 + k, r] = 1
            W[k*4 + i, r] = 1

    # 3. M_4,4,1 (j=0..3, k=4) (16 terms)
    # C_{i,4} = A_{i,j} B_{j,4}
    col_offset2 = 49 + 16
    for i in range(4):
        for j in range(4):
            r = col_offset2 + i*4 + j
            U[i*5 + j, r] = 1
            V[j*5 + 4, r] = 1
            W[4*4 + i, r] = 1 # k=4, so 4*4+i

    # 4. j=4, k=4 (4 terms)
    # C_{i,4} = A_{i,4} B_{4,4}
    col_offset3 = 49 + 16 + 16
    for i in range(4):
        r = col_offset3 + i
        U[i*5 + 4, r] = 1
        V[4*5 + 4, r] = 1
        W[4*4 + i, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


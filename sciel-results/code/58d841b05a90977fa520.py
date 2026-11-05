import numpy as np

def construct():
    n, m, p = 4, 5, 5
    
    # 1. Construct 2x2x2 Strassen decomposition
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    # Indices: i,j in 0..1; j,k in 0..1; k,i in 0..1
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1
    U2[0, 0] = 1; U2[3, 0] = 1 # A00, A11 (idx 0, 3)
    V2[0, 0] = 1; V2[3, 0] = 1 # B00, B11 (idx 0, 3)
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11 (idx 0, 3)
    
    # r=1: M2
    U2[2, 1] = 1; U2[3, 1] = 1 # A10, A11 (idx 2, 3)
    V2[0, 1] = 1 # B00 (idx 0)
    W2[1, 1] = 1; W2[3, 1] = -1 # C10 (k=0,i=1 -> idx 1), C11 (k=1,i=1 -> idx 3)
    
    # r=2: M3
    U2[0, 2] = 1 # A00
    V2[2, 2] = 1; V2[3, 2] = -1 # B10, B11 (idx 2, 3)
    W2[2, 2] = 1; W2[3, 2] = 1 # C01 (k=1,i=0 -> idx 2), C11 (k=1,i=1 -> idx 3)
    
    # r=3: M4
    U2[3, 3] = 1 # A11
    V2[1, 3] = 1; V2[0, 3] = -1 # B01, B00 (idx 1, 0)
    W2[0, 3] = 1; W2[1, 3] = 1 # C00 (idx 0), C10 (idx 1)
    
    # r=4: M5
    U2[0, 4] = 1; U2[1, 4] = 1 # A00, A01 (idx 0, 1)
    V2[3, 4] = 1 # B11
    W2[0, 4] = -1; W2[2, 4] = 1 # C00, C01
    
    # r=5: M6
    U2[2, 5] = 1; U2[0, 5] = -1 # A10, A00
    V2[0, 5] = 1; V2[1, 5] = 1 # B00, B01
    W2[3, 5] = 1 # C11
    
    # r=6: M7
    U2[1, 6] = 1; U2[3, 6] = -1 # A01, A11
    V2[2, 6] = 1; V2[3, 6] = 1 # B10, B11
    W2[0, 6] = 1 # C00

    # 2. Kronecker product to get 4x4x4
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for r1 in range(7):
        for r0 in range(7):
            r = r1 * 7 + r0
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            i_idx = i1 * 4 + i0 # Wait, local 4x4 index is i1*2+i0? No.
                            # In 4x4, index is i. i = i1*2 + i0.
                            # But U4 is indexed by i, j. i in 0..3, j in 0..3.
                            # So row index is i * 4 + j? No, U is (n*m, rank).
                            # n=4, m=4. Row index = i * 4 + j.
                            # Wait, the task says U shape (n*m, rank).
                            # For 4x4x4, n=4, m=4.
                            # i in 0..3, j in 0..3.
                            # In Kronecker, i = i1*2 + i0, j = j1*2 + j0.
                            # So row index = i * 4 + j = (i1*2+i0)*4 + (j1*2+j0).
                            
                            i_val = i1 * 2 + i0
                            j_val = j1 * 2 + j0
                            u_idx = i_val * 4 + j_val
                            U4[u_idx, r] = U2[i1*2+j1, r1] * U2[i0*2+j0, r0]
                            
                            # V4: j, k. j = j1*2+j0, k = k1*2+k0
                            # Row index = j * 4 + k
                            for k1 in range(2):
                                for k0 in range(2):
                                    k_val = k1 * 2 + k0
                                    v_idx = j_val * 4 + k_val
                                    V4[v_idx, r] = V2[j1*2+k1, r1] * V2[j0*2+k0, r0]
                            
                            # W4: k, i. k = k1*2+k0, i = i1*2+i0
                            # Row index = k * 4 + i
                            for k1 in range(2):
                                for k0 in range(2):
                                    k_val = k1 * 2 + k0
                                    w_idx = k_val * 4 + i_val
                                    W4[w_idx, r] = W2[k1*2+i1, r1] * W2[k0*2+i0, r0]

    # 3. Extend to 4x5x5
    # Rank 58
    U = np.zeros((20, 58), dtype=int) # n*m = 4*5 = 20
    V = np.zeros((25, 58), dtype=int) # m*p = 5*5 = 25
    W = np.zeros((20, 58), dtype=int) # p*n = 5*4 = 20
    
    # Part 1: 4x4x4 block (ranks 0..48)
    # U4 indices: i in 0..3, j in 0..3. Global U index: i*5 + j.
    for i in range(4):
        for j in range(4):
            U[i*5 + j, 0:49] = U4[i*4 + j, :]
            
    # V4 indices: j in 0..3, k in 0..3. Global V index: j*5 + k.
    for j in range(4):
        for k in range(4):
            V[j*5 + k, 0:49] = V4[j*4 + k, :]
            
    # W4 indices: k in 0..3, i in 0..3. Global W index: k*4 + i.
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, :]

    # Part 2: A_left * B_12 (ranks 49..52)
    # Contributes to C[i, 4] for i in 0..3, using A[i, j] B[j, 4] for j in 0..3
    for j in range(4):
        r = 49 + j
        # U[i, j', r] = 1 if j' == j
        for i in range(4):
            U[i*5 + j, r] = 1
        # V[j', k', r] = 1 if j' == j and k' == 4
        V[j*5 + 4, r] = 1
        # W[k', i', r] = 1 if k' == 4
        for i_prime in range(4):
            W[4*4 + i_prime, r] = 1

    # Part 3: A_right * B_21 (ranks 53..56)
    # Contributes to C[i, k] for k in 0..3, using A[i, 4] B[4, k]
    for k in range(4):
        r = 53 + k
        # U[i, j', r] = 1 if j' == 4
        for i in range(4):
            U[i*5 + 4, r] = 1
        # V[j', k', r] = 1 if j' == 4 and k' == k
        V[4*5 + k, r] = 1
        # W[k', i', r] = 1 if k' == k
        for i_prime in range(4):
            W[k*4 + i_prime, r] = 1

    # Part 4: A_right * B_22 (rank 57)
    # Contributes to C[i, 4] using A[i, 4] B[4, 4]
    r = 57
    for i in range(4):
        U[i*5 + 4, r] = 1
    V[4*5 + 4, r] = 1
    for i_prime in range(4):
        W[4*4 + i_prime, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

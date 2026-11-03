import numpy as np

def construct():
    # Strassen decomposition for 2x2x2
    # U1, V1, W1 shapes: (4, 7)
    # Indices: A[i, j] -> i*2 + j
    # B[j, k] -> j*2 + k
    # C[i, k] -> i*2 + k
    
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # M1
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # M2
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = 1; W1[3, 1] = -1
    
    # M3
    U1[0, 2] = 1
    V1[2, 2] = 1; V1[3, 2] = -1
    W1[1, 2] = 1; W1[3, 2] = 1
    
    # M4
    U1[3, 3] = 1
    V1[1, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[2, 3] = 1
    
    # M5
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[1, 4] = 1
    
    # M6
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[2, 5] = 1
    W1[3, 5] = 1
    
    # M7
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[3, 6] = 1; V1[1, 6] = 1
    W1[0, 6] = 1; W1[1, 6] = 1

    # 1x2x3 decomposition
    # U2, V2, W2 shapes: (2, 6), (6, 6), (3, 6)
    # A[0, j] -> j (0,1)
    # B[j, k] -> j*3 + k (0..5)
    # C[0, k] -> k (0,1,2)
    
    U2 = np.zeros((2, 6))
    V2 = np.zeros((6, 6))
    W2 = np.zeros((3, 6))
    
    # C0 = A0 B00 + A1 B10
    # Term 1: A0 B00 C0
    U2[0, 0] = 1
    V2[0, 0] = 1 # B[0,0]
    W2[0, 0] = 1
    # Term 2: A1 B10 C0
    U2[1, 1] = 1
    V2[3, 1] = 1 # B[1,0]
    W2[0, 1] = 1
    
    # C1 = A0 B01 + A1 B11
    # Term 3: A0 B01 C1
    U2[0, 2] = 1
    V2[1, 2] = 1 # B[0,1]
    W2[1, 2] = 1
    # Term 4: A1 B11 C1
    U2[1, 3] = 1
    V2[4, 3] = 1 # B[1,1]
    W2[1, 3] = 1
    
    # C2 = A0 B02 + A1 B12
    # Term 5: A0 B02 C2
    U2[0, 4] = 1
    V2[2, 4] = 1 # B[0,2]
    W2[2, 4] = 1
    # Term 6: A1 B12 C2
    U2[1, 5] = 1
    V2[5, 5] = 1 # B[1,2]
    W2[2, 5] = 1

    # Construct 2x4x6 decomposition
    # Rank 42
    U = np.zeros((8, 42))
    V = np.zeros((24, 42))
    W = np.zeros((12, 42))
    
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            
            # U: A is 2x4. Index i*4 + j.
            # j = j1*2 + j2.
            # u1 index: i*2 + j1.
            # u2 index: j2.
            # Total index: (i*2 + j1)*2 + j2 = i*4 + j1*2 + j2.
            # This matches the Kronecker product index if we treat u1 as slow and u2 as fast.
            # u1 has 4 entries. u2 has 2 entries.
            # u_total = u1[r1] (length 4) kron u2[r2] (length 2)?
            # Index k_total = k1 * 2 + k2.
            # k1 = i*2 + j1. k2 = j2.
            # k_total = (i*2 + j1)*2 + j2 = i*4 + j1*2 + j2.
            # Correct.
            
            u1_vec = U1[:, r1]
            u2_vec = U2[:, r2]
            U[:, r] = np.kron(u1_vec, u2_vec)
            
            # V: B is 4x6. Index j*6 + k.
            # j = j1*2 + j2. k = k1*3 + k2.
            # v1 index: j1*2 + k1.
            # v2 index: j2*3 + k2.
            # v1 has 4 entries. v2 has 6 entries.
            # Kronecker index: l1 * 6 + l2.
            # l1 = j1*2 + k1. l2 = j2*3 + k2.
            # l_total = (j1*2 + k1)*6 + (j2*3 + k2) = 12*j1 + 6*k1 + 3*j2 + k2.
            # Required index: j*6 + k = (j1*2 + j2)*6 + (k1*3 + k2) = 12*j1 + 6*j2 + 3*k1 + k2.
            # Mismatch.
            # We need to permute the entries of v2 or v1?
            # The tensor product structure is fixed. The mapping of indices to physical matrix positions is what we control by filling the array.
            # We must map the logical indices (j1, k1, j2, k2) to the physical linear index.
            
            # Let's iterate through non-zeros in v1 and v2
            v1_vec = V1[:, r1]
            v2_vec = V2[:, r2]
            
            # v1 indices: 0..3 corresponding to (j1, k1)
            # j1 = idx // 2, k1 = idx % 2
            # v2 indices: 0..5 corresponding to (j2, k2)
            # j2 = idx // 3, k2 = idx % 3
            
            for idx1 in range(4):
                val1 = v1_vec[idx1]
                if val1 == 0: continue
                j1 = idx1 // 2
                k1 = idx1 % 2
                
                for idx2 in range(6):
                    val2 = v2_vec[idx2]
                    if val2 == 0: continue
                    j2 = idx2 // 3
                    k2 = idx2 % 3
                    
                    j = j1 * 2 + j2
                    k = k1 * 3 + k2
                    lin_idx = j * 6 + k
                    V[lin_idx, r] += val1 * val2
                    
            # W: C is 2x6. Index k*2 + i?
            # Task says W[k*n + i, r]. n=2.
            # So index is k*2 + i.
            # w1 index: i*2 + k1? No, w1 is for 2x2x2. C is 2x2.
            # w1 index: i*2 + k1.
            # w2 index: k2 (since C is 1x3, index is k).
            # w1 has 4 entries. w2 has 3 entries.
            # Kronecker index: m1 * 3 + m2.
            # m1 = i*2 + k1. m2 = k2.
            # m_total = (i*2 + k1)*3 + k2 = 6*i + 3*k1 + k2.
            # Required index: k*2 + i.
            # k = k1*3 + k2.
            # k*2 + i = (k1*3 + k2)*2 + i = 6*k1 + 2*k2 + i.
            # Mismatch again.
            
            w1_vec = W1[:, r1]
            w2_vec = W2[:, r2]
            
            for idx1 in range(4):
                val1 = w1_vec[idx1]
                if val1 == 0: continue
                i = idx1 // 2
                k1 = idx1 % 2
                
                for idx2 in range(3):
                    val2 = w2_vec[idx2]
                    if val2 == 0: continue
                    k2 = idx2
                    
                    i_total = i
                    k_total = k1 * 3 + k2
                    lin_idx = k_total * 2 + i_total
                    W[lin_idx, r] += val1 * val2

    return {'U': U, 'V': V, 'W': W}

import numpy as np

def construct():
    # Strassen algorithm for 2x2x2
    # U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # W[k*2+i]. C10 is k=0,i=1 -> idx 2. C11 is k=1,i=1 -> idx 3.
    
    # M3: A00(B01-B11) -> C01, C11
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01 is k=1,i=0 -> idx 1. C11 is k=1,i=1 -> idx 3.
    
    # M4: A11(B10-B00) -> C00, C10
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00 is k=0,i=0 -> idx 0. C10 is k=0,i=1 -> idx 2.
    
    # M5: (A00+A01)B11 -> -C00, C01
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00 is idx 0. C01 is idx 1.
    
    # M6: (A10-A00)(B00+B01) -> C11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11 is idx 3.
    
    # M7: (A01-A11)(B10+B11) -> C00
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00 is idx 0.

    # Kronecker product for 4x4x4
    # U4 = U2 kron U2? 
    # The tensor T = T1 x T2.
    # U indices (i1, i2, j1, j2). Flattened: (i1*2+i2)*4 + (j1*2+j2)?
    # The task uses row-major flattening: i*m + j.
    # For 4x4, i is 0..3, j is 0..3.
    # In Kronecker product of two 2x2 matrices:
    # A = A1 x A2. A[i1*2+i2, j1*2+j2] = A1[i1, j1] * A2[i2, j2].
    # So the index in U4 for A is (i1*2+i2)*4 + (j1*2+j2).
    # Wait, the flattening in the task is `i * m + j`.
    # For 4x4, `i * 4 + j`.
    # If we map (i1, i2) to i and (j1, j2) to j, then i = i1*2 + i2, j = j1*2 + j2.
    # So index = (i1*2 + i2)*4 + (j1*2 + j2).
    
    # We can construct U4 by iterating r1, r2 from 0..6.
    # U4[i1*2+i2, j1*2+j2, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for r1 in range(7):
        for r2 in range(7):
            r4 = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1 * 2 + j1, r1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2 * 2 + j2, r2]
                            if u2 == 0: continue
                            i4 = i1 * 2 + i2
                            j4 = j1 * 2 + j2
                            U4[i4 * 4 + j4, r4] = u1 * u2
                            
                            v1 = V2[j1 * 2 + (i2), r1] # V indices are (j, k). In 2x2, j is row of B? 
                            # Wait, V indices in 2x2 are (j, k) where j is col of A / row of B.
                            # In Kronecker, B = B1 x B2. B[j1*2+j2, k1*2+k2] = B1[j1, k1] B2[j2, k2].
                            # So V4 indices are (j1*2+j2, k1*2+k2).
                            # V2 index is (j1, k1) for first factor, (j2, k2) for second.
                            # My V2 definition: V2[j*2+k, r].
                            # So V2[j1*2+k1, r1] and V2[j2*2+k2, r2].
                            
                            # Let's re-verify V2 indices.
                            # M1: B00, B11. V2[0,0]=1, V2[3,0]=1. (0,0) and (1,1). Correct.
                            # M2: B00. V2[0,1]=1. (0,0). Correct.
                            
                            # So:
                            v1 = V2[j1 * 2 + i2, r1] # k1 corresponds to i2? No.
                            # B1[j1, k1]. B2[j2, k2].
                            # The "k" index in the 4x4 case is composed of k1, k2.
                            # The "j" index in the 4x4 case is composed of j1, j2.
                            # In U4, we used i1, i2 for rows and j1, j2 for cols of A.
                            # In V4, we use j1, j2 for rows of B and k1, k2 for cols of B.
                            # But j1, j2 are the same as the column indices of A.
                            # So V4 row index is j4 = j1*2+j2.
                            # V4 col index is k4 = k1*2+k2.
                            
                            # We need to sum over k1, k2? No, V is a factor.
                            # V4[j4*4+k4, r4] = sum? No, it's a product of factors.
                            # V4[j1*2+j2, k1*2+k2, r4] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2].
                            
                            pass # This loop is getting complicated. Let's use np.kron.

    # Using np.kron is safer and faster.
    # U4 = np.kron(U2, U2)
    # But we need to reorder indices.
    # np.kron(U2, U2) has shape (4, 49).
    # Row index in kron: i2*4 + i1? Or i1*4 + i2?
    # np.kron(A, B) where A is 4x7, B is 4x7.
    # Result is 16x49.
    # Row i of result is row i//4 of A and row i%4 of B.
    # So row index = i_A * 4 + i_B.
    # Our desired index is i4 * 4 + j4 = (i1*2+i2)*4 + (j1*2+j2).
    # This is NOT i1*4+i2.
    # We need to permute the rows.
    
    # Let's stick to the explicit loop but optimized.
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for r1 in range(7):
        for r2 in range(7):
            r4 = r1 * 7 + r2
            # U
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1 * 2 + j1, r1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2 * 2 + j2, r2]
                            if u2 == 0: continue
                            i4 = i1 * 2 + i2
                            j4 = j1 * 2 + j2
                            U4[i4 * 4 + j4, r4] = u1 * u2
                            
                            # V
                            # V2 indices: j*2+k.
                            # V4 indices: j4*4+k4 = (j1*2+j2)*4 + (k1*2+k2).
                            # V4[j4, k4, r4] = V2[j1, k1, r1] * V2[j2, k2, r2]
                            # We need to iterate k1, k2.
                            # But V2 is sparse.
                            for k1 in range(2):
                                v1 = V2[j1 * 2 + k1, r1]
                                if v1 == 0: continue
                                for k2 in range(2):
                                    v2 = V2[j2 * 2 + k2, r2]
                                    if v2 == 0: continue
                                    k4 = k1 * 2 + k2
                                    V4[j4 * 4 + k4, r4] = v1 * v2
                                    
                            # W
                            # W2 indices: k*2+i.
                            # W4 indices: k4*4+i4 = (k1*2+k2)*4 + (i1*2+i2).
                            # W4[k4, i4, r4] = W2[k1, i1, r1] * W2[k2, i2, r2]
                            for k1 in range(2):
                                w1 = W2[k1 * 2 + i1, r1]
                                if w1 == 0: continue
                                for k2 in range(2):
                                    w2 = W2[k2 * 2 + i2, r2]
                                    if w2 == 0: continue
                                    k4 = k1 * 2 + k2
                                    W4[k4 * 4 + i4, r4] = w1 * w2

    # Now construct 4x4x8
    # Rank 98
    U = np.zeros((16, 98), dtype=int)
    V = np.zeros((32, 98), dtype=int) # m*p = 4*8 = 32
    W = np.zeros((32, 98), dtype=int) # p*n = 8*4 = 32
    
    # First 49 terms for B1 (cols 0-3)
    U[:, 0:49] = U4
    # V for B1: V indices (j, k) with k in 0..3.
    # V4 has indices (j4, k4) with k4 in 0..3.
    # V[j*8+k, r] = V4[j*4+k, r] for k in 0..3.
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, 0:49]
            
    # W for C1: W indices (k, i) with k in 0..3.
    # W4 has indices (k4, i4) with k4 in 0..3.
    # W[k*4+i, r] = W4[k*4+i, r] for k in 0..3.
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, 0:49]
            
    # Next 49 terms for B2 (cols 4-7)
    U[:, 49:98] = U4
    # V for B2: V indices (j, k) with k in 4..7.
    # V[j*8+k, r] = V4[j*4+(k-4), r-49]
    for j in range(4):
        for k in range(4, 8):
            V[j * 8 + k, 49:98] = V4[j * 4 + (k - 4), 0:49]
            
    # W for C2: W indices (k, i) with k in 4..7.
    # W[k*4+i, r] = W4[(k-4)*4+i, r-49]
    for k in range(4, 8):
        for i in range(4):
            W[k * 4 + i, 49:98] = W4[(k - 4) * 4 + i, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


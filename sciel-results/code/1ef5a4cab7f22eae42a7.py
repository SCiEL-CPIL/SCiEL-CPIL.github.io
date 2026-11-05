import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    # Strassen 2x2 decomposition (Rank 7)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Indices: 0:00, 1:01, 2:10, 3:11
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1; W2[1, 0] = -1; W2[2, 0] = 1
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    # M6
    U2[2, 5] = 1; U2[3, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Schoolbook 3x3 decomposition (Rank 27)
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                col = i * 9 + j * 3 + k
                U3[i * 3 + j, col] = 1
                V3[j * 3 + k, col] = 1
                W3[k * 3 + i, col] = 1

    # Kronecker Product
    n1, m1 = 2, 2
    n2, m2 = 3, 3
    R1 = 7
    R2 = 27
    Rank = R1 * R2
    
    U = np.zeros((36, Rank))
    V = np.zeros((36, Rank))
    W = np.zeros((36, Rank))
    
    for r in range(R1):
        for s in range(R2):
            col = r * R2 + s
            # Iterate over all non-zero entries in U2[:, r] and U3[:, s]
            # Since they are sparse, we can just iterate over all indices
            for i1 in range(4):
                if U2[i1, r] == 0: continue
                for i2 in range(9):
                    if U3[i2, s] == 0: continue
                    # Map indices
                    # i1 is (i, j) in 2x2 -> i=i1//2, j=i1%2
                    # i2 is (i', j') in 3x3 -> i'=i2//3, j'=i2%3
                    i = i1 // 2
                    j = i1 % 2
                    ip = i2 // 3
                    jp = i2 % 3
                    
                    # New index in 6x6
                    # Row = i*3 + ip, Col = j*3 + jp
                    idx = (i * 3 + ip) * 6 + (j * 3 + jp)
                    U[idx, col] += U2[i1, r] * U3[i2, s]
                    
                    # V mapping
                    # V2 index is (j, k) -> j=j1//2, k=j1%2
                    # V3 index is (j', k') -> j'=j2//3, k'=j2%3
                    # We need to iterate V2 and V3 separately or combine loops
            # This nested loop approach for U is fine, but V and W need similar treatment.
            # Let's do it more efficiently by precomputing non-zeros or just full loops since 4x9 is small.
            
    # Re-do with full loops for simplicity and correctness
    U = np.zeros((36, Rank))
    V = np.zeros((36, Rank))
    W = np.zeros((36, Rank))
    
    for r in range(R1):
        for s in range(R2):
            col = r * R2 + s
            for i1 in range(4):
                u_val = U2[i1, r]
                v_val = V2[i1, r]
                w_val = W2[i1, r]
                if u_val == 0 and v_val == 0 and w_val == 0: continue
                
                i1_row = i1 // 2
                i1_col = i1 % 2
                
                for i2 in range(9):
                    u2_val = U3[i2, s]
                    v2_val = V3[i2, s]
                    w2_val = W3[i2, s]
                    if u2_val == 0 and v2_val == 0 and w2_val == 0: continue
                    
                    i2_row = i2 // 3
                    i2_col = i2 % 3
                    
                    # U
                    idx_u = (i1_row * 3 + i2_row) * 6 + (i1_col * 3 + i2_col)
                    U[idx_u, col] += u_val * u2_val
                    
                    # V
                    # V2 index i1 corresponds to (j, k) where j=i1_row, k=i1_col? 
                    # In 2x2, V index is j*2+k.
                    # In 3x3, V index is j'*3+k'.
                    # So V2[i1] has row j=i1//2, col k=i1%2.
                    # V3[i2] has row j'=i2//3, col k'=i2%3.
                    idx_v = (i1_row * 3 + i2_row) * 6 + (i1_col * 3 + i2_col)
                    V[idx_v, col] += v_val * v2_val
                    
                    # W
                    # W2 index i1 corresponds to (k, i) where k=i1//2, i=i1%2?
                    # In 2x2, W index is k*2+i.
                    # So W2[i1] has row k=i1//2, col i=i1%2.
                    # W3[i2] has row k'=i2//3, col i'=i2%3.
                    idx_w = (i1_row * 3 + i2_row) * 6 + (i1_col * 3 + i2_col)
                    W[idx_w, col] += w_val * w2_val

    return {"U": U, "V": V, "W": W}

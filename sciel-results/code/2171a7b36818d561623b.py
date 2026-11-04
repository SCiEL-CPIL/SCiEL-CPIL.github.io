import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen 2x2 factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1
    
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[1, 3] = 1; W2[0, 3] = 1
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[2, 4] = 1; W2[0, 4] = -1
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[0, 5] = 1; W2[3, 5] = 1
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Kronecker product to get 4x4 factors
    # U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # i = 2*i1 + i2, j = 2*j1 + j2
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2 * i1 + i2
                    j = 2 * j1 + j2
                    idx_u = i * 4 + j
                    idx_v = j * 4 + i # Wait, V is (j,k). So j1,j2,k1,k2
                    # V index is (j, k).
                    # W index is (k, i).
                    
                    # We need to fill U4, V4, W4 for all k as well?
                    # No, the tensor is T[i,j,k].
                    # The factors are U[i,j], V[j,k], W[k,i].
                    # The Kronecker structure applies to the indices.
                    
                    # U4 row idx_u corresponds to (i,j).
                    # V4 row idx_v corresponds to (j,k).
                    # W4 row idx_w corresponds to (k,i).
                    
                    # We iterate over all i,j,k?
                    # No, we construct the matrices directly.
                    pass
    
    # Better way:
    # U4[i, j, r1, r2] = U2[i1, j1, r1] * U2[i2, j2, r2]
    # Flatten indices.
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2 * i1 + i2
                    j = 2 * j1 + j2
                    u_row = i * 4 + j
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            U4[u_row, r] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                            
                            # V row is (j, k). We need k.
                            # But V4 is independent of i?
                            # V4[j, k, r] depends only on j, k.
                            # So we can fill V4 in a separate loop or inside this one if we iterate k too.
                            # But V4 does not depend on i.
                            # So we can just compute V4 and W4 separately.
                            
    # Compute V4
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2 * j1 + j2
                    k = 2 * k1 + k2
                    v_row = j * 4 + k
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            V4[v_row, r] = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
                            
    # Compute W4
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2 * k1 + k2
                    i = 2 * i1 + i2
                    w_row = k * 4 + i
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            W4[w_row, r] = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
# EVOLVE-BLOCK-END

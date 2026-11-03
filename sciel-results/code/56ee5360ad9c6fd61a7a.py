import numpy as np

def construct():
    # Dimensions
    n, m, p = 2, 4, 8
    
    # Sub-problem 1: 2x2x2 (Strassen)
    n1, m1, p1 = 2, 2, 2
    rank1 = 7
    U1 = np.zeros((n1 * m1, rank1))
    V1 = np.zeros((m1 * p1, rank1))
    W1 = np.zeros((p1 * n1, rank1))
    
    # Strassen terms for 2x2
    # M1 = (a00 + a11)(b00 + b11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1 # c00
    
    # M2 = (a10 + a11)b00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = 1 # c10
    
    # M3 = a00(b01 - b11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[1, 2] = 1 # c01
    
    # M4 = a11(b10 - b00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[2, 3] = 1 # c10
    
    # M5 = (a00 + a01)b11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1 # c00 (subtracted in C00)
    W1[1, 4] = 1   # c01 (added in C01)
    
    # M6 = (a10 - a00)(b00 + b01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1 # c11
    
    # M7 = (a01 - a11)(b10 + b11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1 # c00
    W1[3, 6] = 1 # c11
    
    # Adjust W1 to match C formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Reset W1 and fill correctly based on output coefficients
    W1 = np.zeros((p1 * n1, rank1))
    # r=0 (M1): C00 += 1, C11 += 1
    W1[0, 0] = 1 # k=0, i=0 -> c00
    W1[3, 0] = 1 # k=1, i=1 -> c11
    
    # r=1 (M2): C10 += 1
    W1[2, 1] = 1 # k=1, i=0 -> c10
    
    # r=2 (M3): C01 += 1
    W1[1, 2] = 1 # k=0, i=1 -> c01
    
    # r=3 (M4): C00 += 1, C10 += 1
    W1[0, 3] = 1 # c00
    W1[2, 3] = 1 # c10
    
    # r=4 (M5): C00 -= 1, C01 += 1
    W1[0, 4] = -1 # c00
    W1[1, 4] = 1  # c01
    
    # r=5 (M6): C11 += 1
    W1[3, 5] = 1 # c11
    
    # r=6 (M7): C00 += 1, C11 += 1
    W1[0, 6] = 1 # c00
    W1[3, 6] = 1 # c11

    # Sub-problem 2: 1x2x4 (Schoolbook)
    n2, m2, p2 = 1, 2, 4
    rank2 = 8
    U2 = np.zeros((n2 * m2, rank2))
    V2 = np.zeros((m2 * p2, rank2))
    W2 = np.zeros((p2 * n2, rank2))
    
    for j2 in range(m2):
        for k2 in range(p2):
            r2 = j2 * p2 + k2
            U2[0 * m2 + j2, r2] = 1
            V2[j2 * p2 + k2, r2] = 1
            W2[k2 * n2 + 0, r2] = 1

    # Combine
    rank = rank1 * rank2
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            # Global indices
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            
                            # Local indices for U, V, W
                            u_idx1 = i1 * m1 + j1
                            v_idx1 = j1 * p1 + k1
                            w_idx1 = k1 * n1 + i1
                            
                            u_idx2 = i2 * m2 + j2
                            v_idx2 = j2 * p2 + k2
                            w_idx2 = k2 * n2 + i2
                            
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            
                            for r1 in range(rank1):
                                u1_val = U1[u_idx1, r1]
                                v1_val = V1[v_idx1, r1]
                                w1_val = W1[w_idx1, r1]
                                if u1_val == 0 and v1_val == 0 and w1_val == 0:
                                    continue
                                for r2 in range(rank2):
                                    r = r1 * rank2 + r2
                                    u2_val = U2[u_idx2, r2]
                                    v2_val = V2[v_idx2, r2]
                                    w2_val = W2[w_idx2, r2]
                                    
                                    U[u_idx, r] = u1_val * u2_val
                                    V[v_idx, r] = v1_val * v2_val
                                    W[w_idx, r] = w1_val * w2_val

    # Convert to lists of lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Strassen decomposition for 2x2x2 matrix multiplication
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # A indices: 00(0), 01(1), 10(2), 11(3)
    # B indices: 00(0), 01(1), 10(2), 11(3)
    # C indices (trace convention k*2+i): C00(0), C10(1), C01(2), C11(3)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h) -> contributes to C00, C10, C01, C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 1; W2[2, 0] = 1; W2[3, 0] = 1
    
    # M2 = (c+d)(e+h) -> contributes to C10
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1; V2[3, 1] = 1
    W2[1, 1] = 1
    
    # M3 = (a+b)(h-i) -> contributes to C01
    U2[0, 2] = 1; U2[1, 2] = 1
    V2[3, 2] = 1; V2[2, 2] = -1
    W2[2, 2] = 1
    
    # M4 = (a+b)(i-j) -> contributes to C11
    U2[0, 3] = 1; U2[1, 3] = 1
    V2[2, 3] = 1; V2[3, 3] = -1
    W2[3, 3] = 1
    
    # M5 = (c+d)(f-g) -> contributes to C10
    U2[2, 4] = 1; U2[3, 4] = 1
    V2[1, 4] = 1; V2[0, 4] = -1
    W2[1, 4] = 1
    
    # M6 = (a-b)(e-g) -> contributes to C00
    U2[0, 5] = 1; U2[2, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = -1
    W2[0, 5] = 1
    
    # M7 = (c-a)(h-f) -> contributes to C00
    U2[2, 6] = 1; U2[0, 6] = -1
    V2[3, 6] = 1; V2[1, 6] = -1
    W2[0, 6] = 1
    
    # Trivial decomposition for 1x2x3 matrix multiplication
    # U: (1, 6), V: (6, 6), W: (3, 6)
    # A has 1 element, B has 2x3=6 elements, C has 1x3=3 elements
    # C[k] = sum_j A[0] * B[j, k]
    
    U1 = np.zeros((1, 6))
    V1 = np.zeros((6, 6))
    W1 = np.zeros((3, 6))
    
    # For each (j, k) pair in B, create a term
    # j in 0..1, k in 0..2
    for j in range(2):
        for k in range(3):
            r = j * 3 + k
            U1[0, r] = 1
            V1[j * 3 + k, r] = 1  # B index: j*3 + k
            W1[k, r] = 1           # C index: k (trace convention for n=1: k*1+0 = k)
    
    # Kronecker product: total rank = 7 * 6 = 42
    U_total = np.zeros((8, 42))
    V_total = np.zeros((24, 42))
    W_total = np.zeros((12, 42))
    
    for r2 in range(7):
        for r1 in range(6):
            r = r2 * 6 + r1
            
            # U: A[i, j] where i in 0..1, j in 0..3
            # A[i, j] = A2[i, j2] * A1[0, j1] where j = j2*2 + j1
            for i2 in range(2):
                for j2 in range(2):
                    for j1 in range(2):
                        a_idx = i2 * 4 + j2 * 2 + j1
                        u2_val = U2[i2 * 2 + j2, r2]
                        u1_val = U1[0, r1]
                        U_total[a_idx, r] += u2_val * u1_val
            
            # V: B[j, k] where j in 0..3, k in 0..5
            # B[j, k] = B2[j2, k2] * B1[j1, k1] where j = j2*2 + j1, k = k2*3 + k1
            for j2 in range(2):
                for j1 in range(2):
                    for k2 in range(2):
                        for k1 in range(3):
                            j = j2 * 2 + j1
                            k = k2 * 3 + k1
                            b_idx = j * 6 + k
                            v2_val = V2[j2 * 2 + k2, r2]
                            v1_val = V1[j1 * 3 + k1, r1]
                            V_total[b_idx, r] += v2_val * v1_val
            
            # W: C[i, k] where i in 0..1, k in 0..5
            # C[i, k] = C2[i, k2] * C1[0, k1] where k = k2*3 + k1
            # Trace convention: W[k*n+i, r] = W[k*2+i, r]
            for i2 in range(2):
                for k2 in range(2):
                    for k1 in range(3):
                        k = k2 * 3 + k1
                        w_idx = k * 2 + i2
                        w2_val = W2[k2 * 2 + i2, r2]
                        w1_val = W1[k1, r1]
                        W_total[w_idx, r] += w2_val * w1_val
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

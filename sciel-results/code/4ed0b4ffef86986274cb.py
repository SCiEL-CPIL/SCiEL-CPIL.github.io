import numpy as np

def construct():
    n, m, p = 3, 4, 6
    R = 24
    
    U = np.zeros((n * m, R), dtype=np.int64)
    V = np.zeros((m * p, R), dtype=np.int64)
    W = np.zeros((p * n, R), dtype=np.int64)
    
    # Decomposition parameters
    # M1: 1x4x3 (Rank 4)
    # M2: 3x1x2 (Rank 6)
    
    # Iterate over all rank terms
    for r1 in range(4):
        for r2 in range(6):
            r = r1 * 6 + r2
            
            # M1 factors (for index j)
            # U1: (1*4, 4) -> U1[j, r1] = 1 if j == r1
            # V1: (4*3, 4) -> V1[j*3+k1, r1] = 1 if j == r1
            # W1: (3*1, 4) -> W1[k1*1+i1, r1] = 1 (i1=0)
            
            # M2 factors (for index i, k2)
            # r2 corresponds to (i2, k2) pair
            # i2 in 0..2, k2 in 0..1
            i2 = r2 // 2
            k2 = r2 % 2
            
            # U2: (3*1, 6) -> U2[i2*1+j2, r2] = 1 if i2 == i2_r2 (j2=0)
            # V2: (1*2, 6) -> V2[j2*2+k2, r2] = 1 if k2 == k2_r2 (j2=0)
            # W2: (2*3, 6) -> W2[k2*3+i2, r2] = 1 if k2==k2_r2 and i2==i2_r2

            # Fill U
            # Global i in 0..2, j in 0..3
            # u = i*4 + j
            # U[u, r] = U1[j, r1] * U2[i, r2]
            for i in range(3):
                for j in range(4):
                    u = i * 4 + j
                    u1_val = 1 if j == r1 else 0
                    u2_val = 1 if i == i2 else 0
                    U[u, r] = u1_val * u2_val
            
            # Fill V
            # Global j in 0..3, k in 0..5
            # v = j*6 + k
            # We need to map k to k1, k2.
            # Assumption: k = k1 * 2 + k2? Or k1 + 3*k2?
            # Let's try k1 = k % 3, k2 = k // 3?
            # If k1 in 0..2 and k2 in 0..1.
            # V[v, r] = V1[j*3+k1, r1] * V2[0*2+k2, r2]
            for j in range(4):
                for k in range(6):
                    v = j * 6 + k
                    # Mapping k to k1, k2
                    # Since M1 has dim 3 for k, M2 has dim 2 for k.
                    # Standard Kronecker flattening: k = k1 * 2 + k2?
                    # Let's assume k1 = k % 3, k2 = k // 3.
                    k1 = k % 3
                    k2 = k // 3
                    
                    v1_val = 1 if j == r1 else 0
                    v2_val = 1 if k2 == k2 else 0 # Wait, V2 depends on k2 from r2
                    v2_val = 1 if k2 == (r2 % 2) else 0
                    V[v, r] = v1_val * v2_val
            
            # Fill W
            # Global k in 0..5, i in 0..2
            # w = k*3 + i
            # W[w, r] = W1[k1*1+0, r1] * W2[k2*3+i, r2]
            for k in range(6):
                for i in range(3):
                    w = k * 3 + i
                    k1 = k % 3
                    k2 = k // 3
                    
                    w1_val = 1
                    w2_val = 1 if (k2 == (r2 % 2) and i == (r2 // 2)) else 0
                    W[w, r] = w1_val * w2_val
    
    return {'U': U, 'V': V, 'W': W}

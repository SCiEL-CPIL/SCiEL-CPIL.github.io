import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline for safety
    
    # Base decompositions
    # M_1: 1x1x2, Rank 2
    U1 = np.zeros((1*1, 2), dtype=int)
    V1 = np.zeros((1*2, 2), dtype=int)
    W1 = np.zeros((2*1, 2), dtype=int)
    # c=0: j=0, k=0
    U1[0, 0] = 1
    V1[0, 0] = 1 # j*p+k = 0*2+0=0
    W1[0, 0] = 1 # k*n+i = 0*1+0=0
    # c=1: j=0, k=1
    U1[0, 1] = 1
    V1[1, 1] = 1 # j*p+k = 0*2+1=1
    W1[1, 1] = 1 # k*n+i = 1*1+0=1

    # M_2: 1x2x2, Rank 4
    U2 = np.zeros((1*2, 4), dtype=int)
    V2 = np.zeros((2*2, 4), dtype=int)
    W2 = np.zeros((2*1, 4), dtype=int)
    # c=0: j=0, k=0
    U2[0, 0] = 1 # i*m+j = 0*2+0=0
    V2[0, 0] = 1 # j*p+k = 0*2+0=0
    W2[0, 0] = 1 # k*n+i = 0*1+0=0
    # c=1: j=0, k=1
    U2[0, 1] = 1
    V2[1, 1] = 1 # 0*2+1=1
    W2[1, 1] = 1 # 1*1+0=1
    # c=2: j=1, k=0
    U2[1, 2] = 1 # 0*2+1=1
    V2[2, 2] = 1 # 1*2+0=2
    W2[0, 2] = 1 # 0*1+0=0
    # c=3: j=1, k=1
    U2[1, 3] = 1
    V2[3, 3] = 1 # 1*2+1=3
    W2[1, 3] = 1 # 1*1+0=1

    # M_3: 2x2x2, Rank 7 (Strassen)
    U3 = np.zeros((2*2, 7), dtype=int)
    V3 = np.zeros((2*2, 7), dtype=int)
    W3 = np.zeros((2*2, 7), dtype=int)
    
    # Indices:
    # U: i*2+j. 0:0,0 1:0,1 2:1,0 3:1,1
    # V: j*2+k. 0:0,0 1:0,1 2:1,0 3:1,1
    # W: k*2+i. 0:0,0 1:1,0 2:0,1 3:1,1
    
    # M1: (a+d)(e+h)
    # U: a(0), d(3). V: e(0), h(3).
    U3[0, 0] = 1; U3[3, 0] = 1
    V3[0, 0] = 1; V3[3, 0] = 1
    # W: C00(0)+1, C11(3)+1
    W3[0, 0] = 1; W3[3, 0] = 1
    
    # M2: (c+d)e
    # U: c(2), d(3). V: e(0).
    U3[2, 1] = 1; U3[3, 1] = 1
    V3[0, 1] = 1
    # W: C10(2)+1, C11(3)-1
    W3[2, 1] = 1; W3[3, 1] = -1
    
    # M3: a(f-h)
    # U: a(0). V: f(1), h(3).
    U3[0, 2] = 1
    V3[1, 2] = 1; V3[3, 2] = -1
    # W: C01(1)+1, C11(3)+1
    W3[1, 2] = 1; W3[3, 2] = 1
    
    # M4: d(g-e)
    # U: d(3). V: g(2), e(0).
    U3[3, 3] = 1
    V3[2, 3] = 1; V3[0, 3] = -1
    # W: C00(0)+1, C10(2)+1
    W3[0, 3] = 1; W3[2, 3] = 1
    
    # M5: (a+b)h
    # U: a(0), b(1). V: h(3).
    U3[0, 4] = 1; U3[1, 4] = 1
    V3[3, 4] = 1
    # W: C00(0)-1, C01(1)+1
    W3[0, 4] = -1; W3[1, 4] = 1
    
    # M6: (c-d)(f+g)
    # U: c(2), d(3). V: f(1), g(2).
    U3[2, 5] = 1; U3[3, 5] = -1
    V3[1, 5] = 1; V3[2, 5] = 1
    # W: C11(3)+1
    W3[3, 5] = 1
    
    # M7: (b-c)(e+g)
    # U: b(1), c(2). V: e(0), g(2).
    U3[1, 6] = 1; U3[2, 6] = -1
    V3[0, 6] = 1; V3[2, 6] = 1
    # W: C00(0)+1
    W3[0, 6] = 1

    # Combine M3 (2x2x2), M2 (1x2x2), M1 (1x1x2)
    # Target: 2x4x8
    # n=2, m=4, p=8
    # T = T3 (2,2,2) x T2 (1,2,2) x T1 (1,1,2)
    # n = 2*1*1 = 2
    # m = 2*2*1 = 4
    # p = 2*2*2 = 8
    
    # Rank = 7 * 4 * 2 = 56
    R = 7 * 4 * 2
    U = np.zeros((2*4, R), dtype=int)
    V = np.zeros((4*8, R), dtype=int)
    W = np.zeros((8*2, R), dtype=int)
    
    # Indices for T3: i3, j3, k3
    # Indices for T2: i2, j2, k2
    # Indices for T1: i1, j1, k1
    
    # i = i3*1*1 + i2*1 + i1 ?
    # i = i3 * (n2*n1) + i2 * n1 + i1
    # n3=2, n2=1, n1=1.
    # i = i3 * 1 + i2 * 1 + i1.
    # Since n2=1, i2 is always 0. n1=1, i1 is always 0.
    # So i = i3.
    
    # m3=2, m2=2, m1=1.
    # j = j3 * (m2*m1) + j2 * m1 + j1
    # j = j3 * 2 + j2 * 1 + j1.
    # j1 is always 0.
    # j = 2*j3 + j2.
    
    # p3=2, p2=2, p1=2.
    # k = k3 * (p2*p1) + k2 * p1 + k1
    # k = k3 * 4 + k2 * 2 + k1.
    
    # U index: i*m + j = i*4 + j
    # V index: j*p + k = j*8 + k
    # W index: k*n + i = k*2 + i
    
    # Iterate over all non-zero entries in the base tensors
    # We can just iterate over all indices and multiply values.
    
    # To be efficient, we pre-extract non-zeros or just loop.
    # Loops are small.
    
    for c3 in range(7):
        for c2 in range(4):
            for c1 in range(2):
                c = c3 * 8 + c2 * 2 + c1
                # Accumulate contributions
                # U
                for i3 in range(2):
                    for j3 in range(2):
                        u3_val = U3[i3*2+j3, c3]
                        if u3_val == 0: continue
                        for i2 in range(1):
                            for j2 in range(2):
                                u2_val = U2[i2*2+j2, c2]
                                if u2_val == 0: continue
                                for i1 in range(1):
                                    for j1 in range(1):
                                        u1_val = U1[i1*1+j1, c1]
                                        if u1_val == 0: continue
                                        
                                        i = i3 * 1 + i2 * 1 + i1
                                        j = j3 * 2 + j2 * 1 + j1
                                        
                                        u_idx = i * 4 + j
                                        val = u3_val * u2_val * u1_val
                                        U[u_idx, c] += val
                                        
                                        # V
                                        for k3 in range(2):
                                            v3_val = V3[j3*2+k3, c3]
                                            if v3_val == 0: continue
                                            for k2 in range(2):
                                                v2_val = V2[j2*2+k2, c2]
                                                if v2_val == 0: continue
                                                for k1 in range(2):
                                                    v1_val = V1[j1*2+k1, c1]
                                                    if v1_val == 0: continue
                                                    
                                                    k = k3 * 4 + k2 * 2 + k1
                                                    
                                                    v_idx = j * 8 + k
                                                    val_v = v3_val * v2_val * v1_val
                                                    V[v_idx, c] += val_v
                                                    
                                                    # W
                                                    w3_val = W3[k3*2+i3, c3]
                                                    if w3_val == 0: continue
                                                    w2_val = W2[k2*1+i2, c2]
                                                    if w2_val == 0: continue
                                                    w1_val = W1[k1*1+i1, c1]
                                                    if w1_val == 0: continue
                                                    
                                                    w_idx = k * 2 + i
                                                    val_w = w3_val * w2_val * w1_val
                                                    W[w_idx, c] += val_w

    return {"U": U, "V": V, "W": W}

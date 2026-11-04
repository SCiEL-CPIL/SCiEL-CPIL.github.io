"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit

    n, m, p = DIMENSIONS
    
    # 1. Define Strassen decomposition for 2x2 matrices
    # U2 shape: (2*2, 7) -> indices i*2+j
    # V2 shape: (2*2, 7) -> indices j*2+k
    # W2 shape: (2*2, 7) -> indices k*2+i
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
    # A: a(A00), d(A11)
    U2[0, 0] = 1  # A00
    U2[3, 0] = 1  # A11 (index 1*2+1)
    # B: e(B00), h(B11)
    V2[0, 0] = 1  # B00
    V2[3, 0] = 1  # B11 (index 1*2+1)
    # C: C00 (k=0, i=0 -> 2*0+0=0), C11 (k=1, i=1 -> 2*1+1=3)
    W2[0, 0] = 1
    W2[3, 0] = 1
    
    # r=1: M2 = (c+d)e -> C10 += 1, C11 -= 1
    # A: c(A10), d(A11)
    U2[2, 1] = 1  # A10 (index 1*2+0)
    U2[3, 1] = 1  # A11
    # B: e(B00)
    V2[0, 1] = 1
    # C: C10 (k=0, i=1 -> 2*0+1=1), C11 (k=1, i=1 -> 3)
    W2[1, 1] = 1
    W2[3, 1] = -1
    
    # r=2: M3 = a(f-h) -> C01 += 1, C11 += 1
    # A: a(A00)
    U2[0, 2] = 1
    # B: f(B01), -h(B11)
    V2[1, 2] = 1  # B01 (index 0*2+1)
    V2[3, 2] = -1 # B11
    # C: C01 (k=1, i=0 -> 2*1+0=2), C11 (k=1, i=1 -> 3)
    W2[2, 2] = 1
    W2[3, 2] = 1
    
    # r=3: M4 = d(g-f) -> C00 += 1, C10 += 1
    # A: d(A11)
    U2[3, 3] = 1
    # B: g(B10), -f(B01)
    V2[2, 3] = 1  # B10 (index 1*2+0)
    V2[1, 3] = -1 # B01
    # C: C00 (k=0, i=0 -> 0), C10 (k=0, i=1 -> 1)
    W2[0, 3] = 1
    W2[1, 3] = 1
    
    # r=4: M5 = (a+b)h -> C00 -= 1, C01 += 1
    # A: a(A00), b(A01)
    U2[0, 4] = 1
    U2[1, 4] = 1  # A01 (index 0*2+1)
    # B: h(B11)
    V2[3, 4] = 1
    # C: C00 (0), C01 (2)
    W2[0, 4] = -1
    W2[2, 4] = 1
    
    # r=5: M6 = (c-a)(e+f) -> C11 += 1
    # A: c(A10), -a(A00)
    U2[2, 5] = 1
    U2[0, 5] = -1
    # B: e(B00), f(B01)
    V2[0, 5] = 1
    V2[1, 5] = 1
    # C: C11 (3)
    W2[3, 5] = 1
    
    # r=6: M7 = (b-d)(g+h) -> C00 += 1
    # A: b(A01), -d(A11)
    U2[1, 6] = 1
    U2[3, 6] = -1
    # B: g(B10), h(B11)
    V2[2, 6] = 1
    V2[3, 6] = 1
    # C: C00 (0)
    W2[0, 6] = 1

    # 2. Construct 4x4 decomposition via Kronecker product
    # Rank = 7 * 7 = 49
    rank_2 = 7
    rank_4 = rank_2 * rank_2
    size_2 = 4
    size_4 = 16
    
    U4 = np.zeros((size_4, rank_4), dtype=int)
    V4 = np.zeros((size_4, rank_4), dtype=int)
    W4 = np.zeros((size_4, rank_4), dtype=int)
    
    # Indices for 4x4: i, j in 0..3
    # i = 2*i1 + i2, j = 2*j1 + j2
    # Rank index R = r1 * 7 + r2
    
    for i1 in range(2):
        for i2 in range(2):
            i = 2 * i1 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    # U4[i, j, R] = U2[i1, j1, r1] * U2[i2, j2, r2]
                    for r1 in range(rank_2):
                        u1 = U2[i1 * 2 + j1, r1]
                        if u1 == 0: continue
                        for r2 in range(rank_2):
                            u2 = U2[i2 * 2 + j2, r2]
                            if u2 == 0: continue
                            R = r1 * rank_2 + r2
                            U4[i * 4 + j, R] = u1 * u2
                    
                    # V4[j, k, R] = V2[j1, k1, r1] * V2[j2, k2, r2]
                    # Note: V indices are j, k.
                    # j = 2*j1 + j2, k = 2*k1 + k2
                    for k1 in range(2):
                        for k2 in range(2):
                            k = 2 * k1 + k2
                            for r1 in range(rank_2):
                                v1 = V2[j1 * 2 + k1, r1]
                                if v1 == 0: continue
                                for r2 in range(rank_2):
                                    v2 = V2[j2 * 2 + k2, r2]
                                    if v2 == 0: continue
                                    R = r1 * rank_2 + r2
                                    V4[j * 4 + k, R] = v1 * v2

    # W4[k, i, R] = W2[k1, i1, r1] * W2[k2, i2, r2]
    # W indices are k, i.
    # k = 2*k1 + k2, i = 2*i1 + i2
    for k1 in range(2):
        for k2 in range(2):
            k = 2 * k1 + k2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    for r1 in range(rank_2):
                        w1 = W2[k1 * 2 + i1, r1]
                        if w1 == 0: continue
                        for r2 in range(rank_2):
                            w2 = W2[k2 * 2 + i2, r2]
                            if w2 == 0: continue
                            R = r1 * rank_2 + r2
                            W4[k * 4 + i, R] = w1 * w2

    # Convert to lists for return
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

import numpy as np
import time

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R)
    
    # We will build lists of rows for U, V, W
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Strassen patterns for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U patterns (4x1 vectors, index i*m+j)
    # i=0, j=0 -> 0; i=0, j=1 -> 1; i=1, j=0 -> 2; i=1, j=1 -> 3
    U_pat = [
        [1, 0, 0, 1], # a+d
        [0, 0, 1, 1], # c+d
        [1, 0, 0, 0], # a
        [0, 0, 0, 1], # d
        [1, 1, 0, 0], # a+b
        [-1, 0, 1, 0], # c-a
        [0, 1, 0, -1]  # b-d
    ]
    
    # W patterns (4x1 vectors, index k*n+i)
    # Output C = [[p, q], [r, s]]
    # p = M1 - M4 + M5 - M7
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M3 + M6 + M7
    # Indices: k=0, i=0 -> 0 (p); k=0, i=1 -> 1 (q); k=1, i=0 -> 2 (r); k=1, i=1 -> 3 (s)
    W_pat = [
        [1, 0, 0, 1],   # M1 contributes to p, s
        [0, 0, 1, 0],   # M2 contributes to r
        [0, 1, 0, -1],  # M3 contributes to q, -s
        [-1, 0, 1, 0],  # M4 contributes to -p, r
        [1, 1, 0, 0],   # M5 contributes to p, q
        [0, 0, 0, 1],   # M6 contributes to s
        [-1, 0, 0, 1]   # M7 contributes to -p, s
    ]
    
    # V patterns for a 2x2 block at columns k1, k2
    # B block = [[e, f], [g, h]]
    # Indices in V: j*p + k
    # j=0, k=k1 -> k1; j=0, k=k2 -> k2
    # j=1, k=k1 -> p+k1; j=1, k=k2 -> p+k2
    def get_V_pattern(pat_idx, k1, k2):
        v = [0] * (m * p)
        # pat_idx 0: e+h -> V[0, k1]=1, V[0, k2]=1, V[1, k1]=1, V[1, k2]=1
        if pat_idx == 0:
            v[k1] = 1; v[k2] = 1
            v[p + k1] = 1; v[p + k2] = 1
        elif pat_idx == 1:
            # (c+d)e -> only e
            v[k1] = 1
        elif pat_idx == 2:
            # a(f-h) -> f - h
            v[k2] = 1
            v[p + k2] = -1
        elif pat_idx == 3:
            # d(g-e) -> g - e
            v[p + k1] = 1
            v[k1] = -1
        elif pat_idx == 4:
            # (a+b)h -> h
            v[p + k2] = 1
        elif pat_idx == 5:
            # (c-a)(e+f) -> e+f
            v[k1] = 1; v[k2] = 1
        elif pat_idx == 6:
            # (b-d)(g+h) -> g+h
            v[p + k1] = 1; v[p + k2] = 1
        return v

    # Add 7 Strassen blocks
    for block_idx in range(7):
        k1 = block_idx * 2
        k2 = block_idx * 2 + 1
        for s in range(7):
            U_rows.append(U_pat[s])
            V_rows.append(get_V_pattern(s, k1, k2))
            # W needs to be mapped to the specific block's output columns
            # W index is k*n + i.
            # For this block, output columns are k1, k2.
            # So k in W_pat (0 or 1) maps to k1 or k2.
            w_vec = [0] * (p * n)
            for k_local in range(2):
                for i_local in range(2):
                    val = W_pat[s][k_local * n + i_local]
                    if val != 0:
                        k_global = k1 + k_local
                        w_vec[k_global * n + i_local] = val
            W_rows.append(w_vec)
            
    # Add 1 schoolbook block for 2x2x1
    # Columns k=14
    k1 = 14
    # Terms:
    # C_0, k1 = A_0,0 B_0,k1 + A_0,1 B_1,k1
    # C_1, k1 = A_1,0 B_0,k1 + A_1,1 B_1,k1
    
    # Term 1: A_0,0 B_0,k1 -> C_0, k1
    U_rows.append([1, 0, 0, 0])
    v1 = [0] * (m * p)
    v1[k1] = 1
    V_rows.append(v1)
    w1 = [0] * (p * n)
    w1[k1 * n + 0] = 1
    W_rows.append(w1)
    
    # Term 2: A_0,1 B_1,k1 -> C_0, k1
    U_rows.append([0, 1, 0, 0])
    v2 = [0] * (m * p)
    v2[p + k1] = 1
    V_rows.append(v2)
    w2 = [0] * (p * n)
    w2[k1 * n + 0] = 1
    W_rows.append(w2)
    
    # Term 3: A_1,0 B_0,k1 -> C_1, k1
    U_rows.append([0, 0, 1, 0])
    v3 = [0] * (m * p)
    v3[k1] = 1
    V_rows.append(v3)
    w3 = [0] * (p * n)
    w3[k1 * n + 1] = 1
    W_rows.append(w3)
    
    # Term 4: A_1,1 B_1,k1 -> C_1, k1
    U_rows.append([0, 0, 0, 1])
    v4 = [0] * (m * p)
    v4[p + k1] = 1
    V_rows.append(v4)
    w4 = [0] * (p * n)
    w4[k1 * n + 1] = 1
    W_rows.append(w4)
    
    rank = len(U_rows)
    U = np.array(U_rows).T
    V = np.array(V_rows).T
    W = np.array(W_rows).T
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0 # Safety margin

    n, m, p = DIMENSIONS
    
    # 1. Construct 2x2 Strassen decomposition
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # c=0: M0 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1; W2[2, 0] = -1; W2[1, 0] = -1
    
    # c=1: M1 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # c=2: M2 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # c=3: M3 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # c=4: M4 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # c=5: M5 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # c=6: M6 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # 2. Construct 4x4 decomposition using outer Strassen
    # Outer U, V, W are 4x7 (for 2x2 blocks of 2x2 matrices)
    U_out = U2.copy()
    V_out = V2.copy()
    W_out = W2.copy()
    
    rank = 49
    U4 = np.zeros((16, rank), dtype=int)
    V4 = np.zeros((16, rank), dtype=int)
    W4 = np.zeros((16, rank), dtype=int)
    
    for c_out in range(7):
        for c_in in range(7):
            c = c_out * 7 + c_in
            
            # U4 indices: i (0-3), j (0-3) -> row i*4+j
            # i = 2*p + u, j = 2*q + v
            for p in range(2):
                for q in range(2):
                    coeff_out = U_out[p * 2 + q, c_out]
                    if coeff_out == 0: continue
                    for u in range(2):
                        for v in range(2):
                            coeff_in = U2[u * 2 + v, c_in]
                            if coeff_in == 0: continue
                            row_u = (2 * p + u) * 4 + (2 * q + v)
                            U4[row_u, c] += coeff_out * coeff_in
                            
            # V4 indices: j (0-3), k (0-3) -> row j*4+k
            # j = 2*p + v, k = 2*q + w
            for p in range(2):
                for q in range(2):
                    coeff_out = V_out[p * 2 + q, c_out]
                    if coeff_out == 0: continue
                    for v in range(2):
                        for w in range(2):
                            coeff_in = V2[v * 2 + w, c_in]
                            if coeff_in == 0: continue
                            row_v = (2 * p + v) * 4 + (2 * q + w)
                            V4[row_v, c] += coeff_out * coeff_in
                            
            # W4 indices: k (0-3), i (0-3) -> row k*4+i
            # k = 2*s + w, i = 2*r + u
            for r in range(2):
                for s in range(2):
                    coeff_out = W_out[s * 2 + r, c_out]
                    if coeff_out == 0: continue
                    for u in range(2):
                        for w in range(2):
                            coeff_in = W2[w * 2 + u, c_in]
                            if coeff_in == 0: continue
                            row_w = (2 * s + w) * 4 + (2 * r + u)
                            W4[row_w, c] += coeff_out * coeff_in
                            
    # Verification
    # Check if U4 @ V4.T @ W4.T reconstructs the identity tensor?
    # T_ijk = sum_c U4[i*4+j, c] V4[j*4+k, c] W4[k*4+i, c]
    # Should be 1 if i,j,k valid, 0 otherwise? No, T_ijk=1 for all i,j,k.
    # Wait, the tensor for matrix mult is T_ijk = 1 for all i,j,k.
    # Let's check a few entries.
    
    # Convert to lists for return
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


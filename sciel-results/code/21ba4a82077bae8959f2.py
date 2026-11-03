import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Construct Strassen 2x2x2 (Rank 7)
    # U, V, W are 4x7
    U_222 = np.zeros((4, 7), dtype=int)
    V_222 = np.zeros((4, 7), dtype=int)
    W_222 = np.zeros((4, 7), dtype=int)
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U[i*2+j, r]
    # V[j*2+k, r]
    # W[k*2+i, r]
    
    # M1
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1 # C00, C11 contributions? 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W coefficients for each term r:
    # r=0 (M1): C00 += 1, C11 += 1
    W_222[0, 0] = 1 # k=0, i=0
    W_222[3, 0] = 1 # k=1, i=1
    
    # r=1 (M2): C10 += 1
    U_222[2, 1] = 1; U_222[3, 1] = 1 # A10, A11
    V_222[0, 1] = 1 # B00
    W_222[2, 1] = 1 # k=1, i=0 (C10)
    
    # r=2 (M3): C01 += 1
    U_222[0, 2] = 1 # A00
    V_222[1, 2] = 1; V_222[3, 2] = -1 # B01, -B11
    W_222[1, 2] = 1 # k=0, i=0 (C01)
    
    # r=3 (M4): C00 += 1, C10 += 1
    U_222[3, 3] = 1 # A11
    V_222[2, 3] = 1; V_222[0, 3] = -1 # B10, -B00
    W_222[0, 3] = 1 # C00
    W_222[2, 3] = 1 # C10
    
    # r=4 (M5): C01 += 1, C00 -= 1
    U_222[0, 4] = 1; U_222[1, 4] = 1 # A00, A01
    V_222[3, 4] = 1 # B11
    W_222[1, 4] = 1 # C01
    W_222[0, 4] = -1 # C00
    
    # r=5 (M6): C11 += 1
    U_222[2, 5] = 1; U_222[0, 5] = -1 # A10, -A00
    V_222[0, 5] = 1; V_222[1, 5] = 1 # B00, B01
    W_222[3, 5] = 1 # C11
    
    # r=6 (M7): C00 += 1
    U_222[1, 6] = 1; U_222[3, 6] = -1 # A01, -A11
    V_222[2, 6] = 1; V_222[3, 6] = 1 # B10, B11
    W_222[0, 6] = 1 # C00
    
    # 2. Construct Schoolbook 2x2x1 (Rank 4)
    # n=2, m=2, p=1
    U_221 = np.zeros((4, 4), dtype=int)
    V_221 = np.zeros((2, 4), dtype=int) # j*1+k = j
    W_221 = np.zeros((2, 4), dtype=int) # k*2+i = i
    
    # r=0: A00 B00 -> C00
    U_221[0, 0] = 1; V_221[0, 0] = 1; W_221[0, 0] = 1
    # r=1: A01 B10 -> C00
    U_221[1, 1] = 1; V_221[1, 1] = 1; W_221[0, 1] = 1
    # r=2: A10 B00 -> C10
    U_221[2, 2] = 1; V_221[0, 2] = 1; W_221[1, 2] = 1
    # r=3: A11 B10 -> C10
    U_221[3, 3] = 1; V_221[1, 3] = 1; W_221[1, 3] = 1
    
    # 3. Construct 2x2x3 (Rank 11)
    # Concatenate 222 (k=0,1) and 221 (k=2)
    U_223 = np.zeros((4, 11), dtype=int)
    V_223 = np.zeros((6, 11), dtype=int) # j*3+k
    W_223 = np.zeros((6, 11), dtype=int) # k*2+i
    
    U_223[:, :7] = U_222
    V_223[:, :7] = V_222 # V_222 is 4x7, V_223 is 6x7. Rows 0,1,2,3 map to k=0,1.
    # V_222 rows: j*2+k. j=0,1; k=0,1.
    # V_223 rows: j*3+k.
    # For r < 7:
    # V_223[j*3+k, r] = V_222[j*2+k, r]
    # We need to map indices.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 3
    # j=1, k=1 -> 4
    # V_222 indices:
    # 0, 1, 2, 3 correspond to (0,0), (0,1), (1,0), (1,1)
    # So:
    V_223[0, :7] = V_222[0, :7]
    V_223[1, :7] = V_222[1, :7]
    V_223[3, :7] = V_222[2, :7]
    V_223[4, :7] = V_222[3, :7]
    
    W_223[:, :7] = W_222 # W_222 is 4x7. W_223 is 6x7.
    # W_222 indices: k*2+i.
    # k=0: 0, 1
    # k=1: 2, 3
    # W_223 indices: k*2+i.
    # k=0: 0, 1
    # k=1: 2, 3
    # So direct copy is fine for k=0,1.
    
    # Add 221 part (r=7..10)
    U_223[:, 7:] = U_221
    # V_221 is 2x4 (j). Map to k=2.
    # V_223[j*3+2, 7+r] = V_221[j, r]
    V_223[2, 7:] = V_221[0, :]
    V_223[5, 7:] = V_221[1, :]
    
    # W_221 is 2x4 (i). Map to k=2.
    # W_223[2*2+i, 7+r] = W_221[i, r]
    W_223[4, 7:] = W_221[0, :]
    W_223[5, 7:] = W_221[1, :]
    
    # 4. Construct 4x4x4 (Rank 49) via Kronecker 222 x 222
    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int)
    W_444 = np.zeros((16, 49), dtype=int)
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    i = i1*2 + i2
                    j = j1*2 + j2
                    u_idx = i*4 + j
                    u1_idx = i1*2 + j1
                    u2_idx = i2*2 + j2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            U_444[u_idx, r] = U_222[u1_idx, r1] * U_222[u2_idx, r2]
                            V_444[u_idx, r] = V_222[u1_idx, r1] * V_222[u2_idx, r2]
                            # V index: j*4+k. k = k1*2+k2.
                            # Wait, V_444 index is j*4+k.
                            # V_222 index is j1*2+k1.
                            # We need k.
                            # In the loop above, we didn't loop k.
                            # V_444[j*4+k, r] = V_222[j1*2+k1, r1] * V_222[j2*2+k2, r2]
                            # We need to fill V_444 correctly.
                            pass
                    
    # Redo V_444 and W_444 properly
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    j = j1*2 + j2
                    k = k1*2 + k2
                    v_idx = j*4 + k
                    v1_idx = j1*2 + k1
                    v2_idx = j2*2 + k2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            V_444[v_idx, r] = V_222[v1_idx, r1] * V_222[v2_idx, r2]
                            
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    k = k1*2 + k2
                    i = i1*2 + i2
                    w_idx = k*4 + i
                    w1_idx = k1*2 + i1
                    w2_idx = k2*2 + i2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            W_444[w_idx, r] = W_222[w1_idx, r1] * W_222[w2_idx, r2]
                            
    # 5. Construct 4x4x3 (Rank 44) via Kronecker 223 x 221
    U_443 = np.zeros((16, 44), dtype=int)
    V_443 = np.zeros((12, 44), dtype=int) # j*3+k
    W_443 = np.zeros((12, 44), dtype=int) # k*4+i
    
    # U
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    i = i1*2 + i2
                    j = j1*2 + j2
                    u_idx = i*4 + j
                    u1_idx = i1*2 + j1
                    u2_idx = i2*2 + j2
                    for r1 in range(11):
                        for r2 in range(4):
                            r = r1*4 + r2
                            U_443[u_idx, r] = U_223[u1_idx, r1] * U_221[u2_idx, r2]
                            
    # V
    for j1 in range(2):
        for k1 in range(3):
            for j2 in range(2):
                # k2 is always 0
                j = j1*2 + j2
                k = k1*1 + 0
                v_idx = j*3 + k
                v1_idx = j1*3 + k1
                v2_idx = j2*1 + 0
                for r1 in range(11):
                    for r2 in range(4):
                        r = r1*4 + r2
                        V_443[v_idx, r] = V_223[v1_idx, r1] * V_221[v2_idx, r2]
                        
    # W
    for k1 in range(3):
        for i1 in range(2):
            for k2 in range(1): # k2=0
                for i2 in range(2):
                    k = k1*1 + k2
                    i = i1*2 + i2
                    w_idx = k*4 + i
                    w1_idx = k1*2 + i1
                    w2_idx = k2*2 + i2
                    for r1 in range(11):
                        for r2 in range(4):
                            r = r1*4 + r2
                            W_443[w_idx, r] = W_223[w1_idx, r1] * W_221[w2_idx, r2]
                            
    # 6. Combine to 4x4x7 (Rank 93)
    U_447 = np.zeros((16, 93), dtype=int)
    V_447 = np.zeros((28, 93), dtype=int) # j*7+k
    W_447 = np.zeros((28, 93), dtype=int) # k*4+i
    
    U_447[:, :49] = U_444
    U_447[:, 49:] = U_443
    
    # V_444: j*4+k (k in 0..3)
    # V_447: j*7+k
    # Map V_444 to k=0..3
    for j in range(4):
        for k in range(4):
            V_447[j*7+k, :49] = V_444[j*4+k, :49]
            
    # Map V_443 to k=4..6
    # V_443: j*3+k_local (k_local in 0..2)
    # V_447: j*7+k_global
    for j in range(4):
        for k_local in range(3):
            k_global = k_local + 4
            V_447[j*7+k_global, 49:] = V_443[j*3+k_local, 49:]
            
    # W_444: k*4+i (k in 0..3)
    # W_447: k*4+i
    for k in range(4):
        for i in range(4):
            W_447[k*4+i, :49] = W_444[k*4+i, :49]
            
    # Map W_443 to k=4..6
    # W_443: k_local*4+i
    # W_447: k_global*4+i
    for k_local in range(3):
        k_global = k_local + 4
        for i in range(4):
            W_447[k_global*4+i, 49:] = W_443[k_local*4+i, 49:]
            
    # Convert to lists
    U_list = U_447.tolist()
    V_list = V_447.tolist()
    W_list = W_447.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

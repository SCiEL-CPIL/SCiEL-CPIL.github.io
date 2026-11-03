import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Define Strassen 2x2x2 decomposition
    # U, V, W for rank 7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    strassen_2 = [
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([-1, 1, 0, 0])),
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]))
    ]
    
    # 2. Construct 4x4x4 decomposition (Rank 49)
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    block_coeffs = [
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([-1, 1, 0, 0])),
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]))
    ]
    
    col = 0
    for block_idx in range(7):
        U_b, V_b, W_b = block_coeffs[block_idx]
        
        for scalar_idx in range(7):
            U_s, V_s, W_s = strassen_2[scalar_idx]
            
            for i in range(2):
                for j in range(2):
                    coeff_b = U_b[i*2+j]
                    if coeff_b == 0: continue
                    for li in range(2):
                        for lj in range(2):
                            coeff_s = U_s[li*2+lj]
                            if coeff_s == 0: continue
                            global_idx = (i*2+li)*4 + (j*2+lj)
                            U_4[global_idx, col] += coeff_b * coeff_s
                            
            for j in range(2):
                for k in range(2):
                    coeff_b = V_b[j*2+k]
                    if coeff_b == 0: continue
                    for lj in range(2):
                        for lk in range(2):
                            coeff_s = V_s[lj*2+lk]
                            if coeff_s == 0: continue
                            global_idx = (j*2+lj)*4 + (k*2+lk)
                            V_4[global_idx, col] += coeff_b * coeff_s
                            
            for k in range(2):
                for i in range(2):
                    coeff_b = W_b[k*2+i]
                    if coeff_b == 0: continue
                    for lk in range(2):
                        for li in range(2):
                            coeff_s = W_s[lk*2+li]
                            if coeff_s == 0: continue
                            global_idx = (k*2+lk)*4 + (i*2+li)
                            W_4[global_idx, col] += coeff_b * coeff_s
                            
            col += 1
            
            if time.time() > deadline:
                return {"U": [[0]*98 for _ in range(16)], "V": [[0]*98 for _ in range(32)], "W": [[0]*98 for _ in range(32)]}
        if time.time() > deadline:
            return {"U": [[0]*98 for _ in range(16)], "V": [[0]*98 for _ in range(32)], "W": [[0]*98 for _ in range(32)]}
            
    # 3. Construct 4x4x8 decomposition (Rank 98)
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    U[:, 0:49] = U_4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V_4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W_4[k*4+i, 0:49]
            
    U[:, 49:98] = U_4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V_4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W_4[k*4+i, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import numpy as np
import time

def construct():
    # Dimensions
    n, m, p = 4, 4, 9
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    # Start with the current best known decomposition (rank 114)
    # Strassen 4x4x4 for first two 4x4 blocks, schoolbook for last 4x1 block
    
    U22 = np.array([
        [1, 1, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, -1],
        [1, 0, 0, 1, 0, -1, 0]
    ], dtype=float)
    
    V22 = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 0, 1, 0],
        [1, 1, 0, -1, 0, 1, 0]
    ], dtype=float)
    
    W22 = np.array([
        [1, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, -1, 0],
        [-1, 1, 0, 1, 0, 0, 1]
    ], dtype=float)
    
    U44 = np.kron(U22, U22)
    V44 = np.kron(V22, V22)
    W44 = np.kron(W22, W22)
    
    U41 = np.zeros((16, 16), dtype=float)
    V41 = np.zeros((4, 16), dtype=float)
    W41 = np.zeros((4, 16), dtype=float)
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U41[i * 4 + j, col] = 1.0
            V41[j, col] = 1.0
            W41[i, col] = 1.0
            
    R_base = 49 + 49 + 16
    U_total = np.zeros((16, R_base), dtype=float)
    V_total = np.zeros((36, R_base), dtype=float)
    W_total = np.zeros((36, R_base), dtype=float)
    
    U_total[:, 0:49] = U44
    V_total[0:16, 0:49] = V44
    W_total[0:16, 0:49] = W44
    
    U_total[:, 49:98] = U44
    V_total[16:32, 49:98] = V44
    W_total[16:32, 49:98] = W44
    
    U_total[:, 98:114] = U41
    V_total[32:36, 98:114] = V41
    W_total[32:36, 98:114] = W41
    
    # Verify base decomposition
    T_recon = np.einsum('uc,vc,wc->uvw', U_total, V_total, W_total, optimize=True)
    T_target = np.zeros((16, 36, 36), dtype=float)
    for i in range(4):
        for j in range(4):
            for k in range(9):
                u = i * 4 + j
                v = j * 9 + k
                w = k * 4 + i
                T_target[u, v, w] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook
        R_school = 144
        U_school = np.zeros((16, R_school), dtype=float)
        V_school = np.zeros((36, R_school), dtype=float)
        W_school = np.zeros((36, R_school), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_school[i*4+j, col] = 1.0
                    V_school[j*9+k, col] = 1.0
                    W_school[k*4+i, col] = 1.0
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to improve using a 9x9x9 embedding approach
    # The idea: embed the 4x4x9 problem into a 9x9x9 problem
    # If we have a good 9x9x9 decomposition, we can extract the 4x4x9 part
    
    # For now, return the base decomposition
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

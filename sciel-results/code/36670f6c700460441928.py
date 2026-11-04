import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    # Strassen decomposition for 2x2x2 matrix multiplication
    # C = A * B, where A, B, C are 2x2
    # We define the 7 tensors (u, v, w) such that C = sum u_t * v_t * w_t
    # Indices for 2x2: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    
    # U terms (A coefficients)
    # V terms (B coefficients)
    # W terms (C coefficients)
    
    # T1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    
    # T2: (A10+A11)B00 -> C10 += 1, C11 -= 1
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1]
    
    # T3: A00(B01-B11) -> C01 += 1, C11 += 1
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1]
    
    # T4: A11(B10-B00) -> C00 += 1, C10 += 1
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0]
    
    # T5: (A00+A01)B11 -> C00 -= 1, C01 += 1
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]
    
    # T6: (A10-A00)(B00+B01) -> C11 += 1
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    
    # T7: (A01-A11)(B10+B11) -> C00 += 1
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0]
    
    strassen_u = np.array([u1, u2, u3, u4, u5, u6, u7], dtype=float)
    strassen_v = np.array([v1, v2, v3, v4, v5, v6, v7], dtype=float)
    strassen_w = np.array([w1, w2, w3, w4, w5, w6, w7], dtype=float)
    
    # Global matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Map local 2x2 indices to global indices
    # Local (i, j) in 2x2 block -> Global (i, j0 + j)
    # Local (j, k) in 2x2 block -> Global (j0 + j, k0 + k)
    # Local (k, i) in 2x2 block -> Global (k0 + k, i)
    
    col_offset = 0
    for j_pair in range(2):  # j0 = 0, 2
        for k_pair in range(4):  # k0 = 0, 2, 4, 6
            j0 = j_pair * 2
            k0 = k_pair * 2
            
            for t in range(7):
                col = col_offset + t
                
                # Update U
                for i in range(2):
                    for j in range(2):
                        val = strassen_u[t, i * 2 + j]
                        if val != 0:
                            U[i * m + (j0 + j), col] += val
                            
                # Update V
                for j in range(2):
                    for k in range(2):
                        val = strassen_v[t, j * 2 + k]
                        if val != 0:
                            V[(j0 + j) * p + (k0 + k), col] += val
                            
                # Update W
                for k in range(2):
                    for i in range(2):
                        val = strassen_w[t, k * 2 + i]
                        if val != 0:
                            W[(k0 + k) * n + i, col] += val
                            
            col_offset += 7
            
            if time.time() - _START_TIME > _DEADLINE:
                break
        if time.time() - _START_TIME > _DEADLINE:
            break
            
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))

import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    timeout = 120.0
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (16, 56)
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for A (s) and B (t)
    # s: [s11, s12, s21, s22]
    # t: [t11, t12, t21, t22]
    s_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    t_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Coefficients for C blocks
    # c[term][block]
    # block 0: C11, 1: C12, 2: C21, 3: C22
    c_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    for t in range(7):
        s = s_coeffs[t]
        tc = t_coeffs[t]
        c = c_coeffs[t]
        
        for j in range(2):
            for k in range(4):
                r = t * 8 + j * 4 + k
                
                # Fill U
                # U index: i * m + col = i * 4 + col
                # j is the local column index (0 or 1)
                # A[0, j] is A11[j]
                # A[0, j+2] is A12[j]
                # A[1, j] is A21[j]
                # A[1, j+2] is A22[j]
                
                # A11[j]
                idx = 0 * 4 + j
                U[idx, r] += s[0]
                
                # A12[j]
                idx = 0 * 4 + (j + 2)
                U[idx, r] += s[1]
                
                # A21[j]
                idx = 1 * 4 + j
                U[idx, r] += s[2]
                
                # A22[j]
                idx = 1 * 4 + (j + 2)
                U[idx, r] += s[3]
                
                # Fill V
                # V index: i * p + col = i * 8 + col
                # j is local row (0 or 1)
                # k is local col (0..3)
                # B[j, k] is B11[j, k]
                # B[j, k+4] is B12[j, k]
                # B[j+2, k] is B21[j, k]
                # B[j+2, k+4] is B22[j, k]
                
                # B11[j, k]
                idx = j * 8 + k
                V[idx, r] += tc[0]
                
                # B12[j, k]
                idx = j * 8 + (k + 4)
                V[idx, r] += tc[1]
                
                # B21[j, k]
                idx = (j + 2) * 8 + k
                V[idx, r] += tc[2]
                
                # B22[j, k]
                idx = (j + 2) * 8 + (k + 4)
                V[idx, r] += tc[3]
                
                # Fill W
                # W index: k_global * n + i_global = k_global * 2 + i_global
                # k is local col (0..3)
                # M_t[k] contributes to C[0, k], C[0, 4+k], C[1, k], C[1, 4+k]
                
                # C[0, k]
                idx = k * 2 + 0
                W[idx, r] += c[0]
                
                # C[0, 4+k]
                idx = (k + 4) * 2 + 0
                W[idx, r] += c[1]
                
                # C[1, k]
                idx = k * 2 + 1
                W[idx, r] += c[2]
                
                # C[1, 4+k]
                idx = (k + 4) * 2 + 1
                W[idx, r] += c[3]
                
                if time.time() - start_time > timeout - 5:
                    break
            if time.time() - start_time > timeout - 5:
                break
        if time.time() - start_time > timeout - 5:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

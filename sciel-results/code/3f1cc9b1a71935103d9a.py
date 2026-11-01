import time
import numpy as np

def construct():
    start_time = time.time()
    
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    R = num_blocks * 7  # 56
    
    U = np.zeros((n * m, R), dtype=np.int64)
    V = np.zeros((m * p, R), dtype=np.int64)
    W = np.zeros((p * n, R), dtype=np.int64)
    
    # Strassen definitions for 2x2x2
    # A indices: 0(A11), 1(A12), 2(A21), 3(A22)
    # B local indices: 0(B11), 1(B12), 2(B21), 3(B22)
    # C local indices: 0(C11), 1(C12), 2(C21), 3(C22)
    
    U_coeffs = [
        [1, 0, 0, 1],   # M1: (A11 + A22)
        [0, 0, 1, 1],   # M2: (A21 + A22)
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: (A11 + A12)
        [0, 0, 1, 0],   # M6: A21  (wait, standard Strassen varies)
        [0, 1, 0, 0]    # M7: A12  (wait, let me verify standard Strassen)
    ]
    
    # Let's use the standard Strassen formulas carefully.
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a22 - a12)(b21 + b22)
    
    # C11 = M1 - M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M4
    
    # U (A-side coefficients)
    U_coeffs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6: a21 - a11
        [0, -1, 0, 1]   # M7: a22 - a12
    ]
    
    # V (B-side coefficients)
    V_coeffs = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [0, -1, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    
    # W (C-side coefficients)
    # C11 = M1 - M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M4
    W_coeffs = [
        [1, 0, 0, 0],   # M1 -> C11
        [0, 0, 1, 0],   # M2 -> C21
        [0, 1, 0, 0],   # M3 -> C12
        [-1, 0, 0, 1],  # M4 -> C11(-1), C22(1)
        [0, 1, 0, 0],   # M5 -> C12
        [0, 0, 1, 0],   # M6 -> C21
        [1, 0, 0, -1]   # M7 -> C11(1), C22(-1)
    ]
    
    for q in range(num_blocks):
        # Global indices for B in this block
        b_idx_11 = 2*q
        b_idx_12 = 2*q + 1
        b_idx_21 = 16 + 2*q
        b_idx_22 = 16 + 2*q + 1
        
        # Global indices for C in this block
        c_idx_11 = 4*q
        c_idx_12 = 4*q + 2
        c_idx_21 = 4*q + 1
        c_idx_22 = 4*q + 3
        
        for s in range(7):
            r = q * 7 + s
            
            # Fill U
            U[0, r] = U_coeffs[s][0]
            U[1, r] = U_coeffs[s][1]
            U[2, r] = U_coeffs[s][2]
            U[3, r] = U_coeffs[s][3]
            
            # Fill V
            V[b_idx_11, r] = V_coeffs[s][0]
            V[b_idx_12, r] = V_coeffs[s][1]
            V[b_idx_21, r] = V_coeffs[s][2]
            V[b_idx_22, r] = V_coeffs[s][3]
            
            # Fill W
            W[c_idx_11, r] = W_coeffs[s][0]
            W[c_idx_12, r] = W_coeffs[s][1]
            W[c_idx_21, r] = W_coeffs[s][2]
            W[c_idx_22, r] = W_coeffs[s][3]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n = 6
    m = 6
    p = 6
    
    # Block size 2
    # 3 blocks in each dimension
    # 9 output blocks, each summing over 3 intermediate blocks
    # 27 block multiplications
    # Each block multiplication uses Strassen (rank 7)
    rank = 27 * 7
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for A (U), B (V), C (W)
    # Indices 0..6 for s
    # A coeffs: U[i_local, j_local, s]
    # B coeffs: V[j_local, k_local, s]
    # C coeffs: W[k_local, i_local, s]
    
    # Local indices 0,1 for 2x2 blocks
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    A_coeffs = np.zeros((2, 2, 7))
    B_coeffs = np.zeros((2, 2, 7))
    C_coeffs = np.zeros((2, 2, 7))
    
    # s=0 (M1)
    A_coeffs[0, 0, 0] = 1; A_coeffs[1, 1, 0] = 1
    B_coeffs[0, 0, 0] = 1; B_coeffs[1, 1, 0] = 1
    C_coeffs[0, 0, 0] = 1; C_coeffs[1, 1, 0] = 1
    
    # s=1 (M2)
    A_coeffs[1, 0, 1] = 1; A_coeffs[1, 1, 1] = 1
    B_coeffs[0, 0, 1] = 1
    C_coeffs[1, 0, 1] = 1; C_coeffs[1, 1, 1] = -1
    
    # s=2 (M3)
    A_coeffs[0, 0, 2] = 1
    B_coeffs[0, 1, 2] = 1; B_coeffs[1, 1, 2] = -1
    C_coeffs[0, 1, 2] = 1; C_coeffs[1, 1, 2] = 1
    
    # s=3 (M4)
    A_coeffs[1, 1, 3] = 1
    B_coeffs[1, 0, 3] = 1; B_coeffs[0, 0, 3] = -1
    C_coeffs[0, 0, 3] = 1; C_coeffs[1, 0, 3] = 1
    
    # s=4 (M5)
    A_coeffs[0, 0, 4] = 1; A_coeffs[0, 1, 4] = 1
    B_coeffs[1, 1, 4] = 1
    C_coeffs[0, 0, 4] = -1; C_coeffs[0, 1, 4] = 1
    
    # s=5 (M6)
    A_coeffs[1, 0, 5] = 1; A_coeffs[0, 0, 5] = -1
    B_coeffs[0, 0, 5] = 1; B_coeffs[0, 1, 5] = 1
    C_coeffs[1, 1, 5] = 1
    
    # s=6 (M7)
    A_coeffs[0, 1, 6] = 1; A_coeffs[1, 1, 6] = -1
    B_coeffs[1, 0, 6] = 1; B_coeffs[1, 1, 6] = 1
    C_coeffs[0, 0, 6] = 1
    
    for i_b in range(3):
        for k_b in range(3):
            for j_b in range(3):
                for s in range(7):
                    r = 7 * (i_b * 3 + k_b) * 3 + 7 * j_b + s
                    
                    for i_0 in range(2):
                        for j_0 in range(2):
                            i = 2 * i_b + i_0
                            j = 2 * j_b + j_0
                            U[i * m + j, r] = A_coeffs[i_0, j_0, s]
                            
                    for j_0 in range(2):
                        for k_0 in range(2):
                            j = 2 * j_b + j_0
                            k = 2 * k_b + k_0
                            V[j * p + k, r] = B_coeffs[j_0, k_0, s]
                            
                    for k_0 in range(2):
                        for i_0 in range(2):
                            k = 2 * k_b + k_0
                            i = 2 * i_b + i_0
                            W[k * n + i, r] = C_coeffs[i_0, k_0, s]
                            
    if time.time() > deadline - 10:
        return {'U': U, 'V': V, 'W': W}
        
    return {'U': U, 'V': V, 'W': W}

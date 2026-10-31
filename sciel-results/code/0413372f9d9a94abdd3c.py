import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 12.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen's 7 multiplications for 2x2
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # M0 = (A00 + A11)(B00 + B11)
        [[0, 0], [1, 1]],   # M1 = (A10 + A11)B00
        [[1, 0], [0, 0]],   # M2 = A00(B01 - B11)
        [[0, 0], [0, 1]],   # M3 = A11(B10 - B00)
        [[1, 1], [0, 0]],   # M4 = (A00 + A01)B11
        [[-1, 0], [1, 0]],  # M5 = (A10 - A00)(B00 + B01)
        [[0, 1], [0, -1]],  # M6 = (A01 - A11)(B10 + B11)
    ])
    
    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # (A00+A11)(B00+B11)
        [[1, 0], [0, 0]],   # (A10+A11)B00
        [[0, 1], [0, -1]],  # A00(B01-B11)
        [[-1, 0], [1, 0]],  # A11(B10-B00)
        [[0, 0], [0, 1]],   # (A00+A01)B11
        [[1, 1], [0, 0]],   # (A10-A00)(B00+B01)
        [[0, 0], [1, 1]],   # (A01-A11)(B10+B11)
    ])
    
    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # C00+=M0, C11+=M0
        [[0, 0], [1, -1]],  # C10+=M1, C11-=M1
        [[0, 1], [0, 1]],   # C01+=M2, C11+=M2
        [[1, 0], [1, 0]],   # C00+=M3, C10+=M3
        [[-1, 1], [0, 0]],  # C00-=M4, C01+=M4
        [[0, 0], [0, 1]],   # C11+=M5
        [[1, 0], [0, 0]],   # C00+=M6
    ])
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t
            
            # U[i*4+j, c_idx]: i = 2*S+s2, j = 2*T+t2
            # coeff = a_coeffs[s,S,T] * a_coeffs[t,s2,t2]
            for S in range(2):
                for T in range(2):
                    a_st = a_coeffs[s, S, T]
                    if a_st == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            a_t2 = a_coeffs[t, s2, t2]
                            if a_t2 == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j, c_idx] += a_st * a_t2
            
            # V[j*4+k, c_idx]: j = 2*T+t2, k = 2*K+k2
            # coeff = b_coeffs[s,T,K] * b_coeffs[t,t2,k2]
            for T in range(2):
                for K in range(2):
                    b_st = b_coeffs[s, T, K]
                    if b_st == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            b_t2 = b_coeffs[t, t2, k2]
                            if b_t2 == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k, c_idx] += b_st * b_t2
            
            # W[k*4+i, c_idx]: k = 2*K+k2, i = 2*S+s2
            # coeff = c_coeffs[s,S,K] * c_coeffs[t,s2,k2]
            for K in range(2):
                for S in range(2):
                    c_sk = c_coeffs[s, S, K]
                    if c_sk == 0:
                        continue
                    for k2 in range(2):
                        for s2 in range(2):
                            c_t2 = c_coeffs[t, s2, k2]
                            if c_t2 == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W[k * 4 + i, c_idx] += c_sk * c_t2
            
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

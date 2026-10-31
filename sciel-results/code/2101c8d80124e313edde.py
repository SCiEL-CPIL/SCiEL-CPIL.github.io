import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 4, 4, 4
    
    # Strassen 2x2 decomposition coefficients
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # M1
        [[0, 0], [1, 1]],   # M2
        [[1, 0], [0, 0]],   # M3
        [[0, 0], [0, 1]],   # M4
        [[1, 1], [0, 0]],   # M5
        [[-1, 0], [1, 0]],  # M6
        [[0, 1], [0, -1]],  # M7
    ])
    
    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[0, 1], [0, -1]],  # M3
        [[-1, 0], [1, 0]],  # M4
        [[0, 0], [0, 1]],   # M5
        [[1, 1], [0, 0]],   # M6
        [[0, 0], [1, 1]],   # M7
    ])
    
    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # C11 += M1, C22 += M1
        [[0, 0], [1, -1]],  # C21 += M2, C22 -= M2
        [[0, 1], [0, 1]],   # C12 += M3, C22 += M3
        [[1, 0], [1, 0]],   # C11 += M4, C21 += M4
        [[-1, 1], [0, 0]],  # C11 -= M5, C12 += M5
        [[0, 0], [0, 1]],   # C22 += M6
        [[1, 0], [0, 0]],   # C11 += M7
    ])
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t
            
            # Vectorized update for U
            # U row index is i*m + j
            # U[i*m+j, c_idx] += a_coeffs[s][S][T] * a_coeffs[t][s2][t2]
            # where i = 2*S + s2, j = 2*T + t2
            
            # We can loop over the 4 outer block positions and 4 inner block positions
            for S in range(2):
                for T in range(2):
                    a_out = a_coeffs[s][S][T]
                    if a_out == 0: continue
                    for s2 in range(2):
                        for t2 in range(2):
                            a_in = a_coeffs[t][s2][t2]
                            if a_in == 0: continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * m + j, c_idx] += a_out * a_in
            
            # Vectorized update for V
            for T in range(2):
                for K in range(2):
                    b_out = b_coeffs[s][T][K]
                    if b_out == 0: continue
                    for t2 in range(2):
                        for k2 in range(2):
                            b_in = b_coeffs[t][t2][k2]
                            if b_in == 0: continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * p + k, c_idx] += b_out * b_in
            
            # Vectorized update for W
            for K in range(2):
                for S in range(2):
                    c_out = c_coeffs[s][S][K]
                    if c_out == 0: continue
                    for k2 in range(2):
                        for s2 in range(2):
                            c_in = c_coeffs[t][s2][k2]
                            if c_in == 0: continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W[k * n + i, c_idx] += c_out * c_in
            
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M_s = (sum_{i,j} A_s[i,j] A_{i,j}) * (sum_{j,k} B_s[j,k] B_{j,k})
    # C_{i,k} += C_s[i,k] M_s
    # Indices i, j, k in {0, 1}
    
    # A_s[i][j]
    A_coeffs = [
        [[1, 0], [0, 1]],  # s=0: A00 + A11
        [[0, 0], [1, 1]],  # s=1: A10 + A11
        [[1, 0], [0, 0]],  # s=2: A00
        [[0, 0], [0, 1]],  # s=3: A11
        [[1, 1], [0, 0]],  # s=4: A00 + A01
        [[-1, 0], [1, 0]], # s=5: A10 - A00
        [[0, 1], [0, -1]], # s=6: A01 - A11
    ]
    
    # B_s[j][k]
    B_coeffs = [
        [[1, 0], [0, 1]],  # s=0: B00 + B11
        [[1, 0], [0, 0]],  # s=1: B00
        [[0, 1], [0, -1]], # s=2: B01 - B11
        [[-1, 0], [1, 0]], # s=3: B10 - B00
        [[0, 0], [0, 1]],  # s=4: B11
        [[1, 1], [0, 0]],  # s=5: B00 + B01
        [[0, 0], [1, 1]],  # s=6: B10 + B11
    ]
    
    # C_s[i][k]
    C_coeffs = [
        [[1, 0], [0, 1]],  # s=0: C00 += M, C11 += M
        [[0, 0], [1, -1]], # s=1: C10 += M, C11 -= M
        [[0, 1], [0, 1]],  # s=2: C01 += M, C11 += M
        [[1, 0], [1, 0]],  # s=3: C00 += M, C10 += M
        [[-1, 1], [0, 0]], # s=4: C00 -= M, C01 += M
        [[0, 0], [0, 1]],  # s=5: C11 += M
        [[1, 0], [0, 0]],  # s=6: C00 += M
    ]
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Outer Strassen steps (block level)
    for s in range(7):
        # Inner Strassen steps (element level within blocks)
        for t in range(7):
            r = 7 * s + t
            
            # U construction: U[i*4+j, r]
            # i = 2*I + i', j = 2*J + j'
            # U[i*4+j, r] = A_coeffs[s][I][J] * A_coeffs[t][i'][j']
            for I in range(2):
                for J in range(2):
                    a_s = A_coeffs[s][I][J]
                    if a_s == 0:
                        continue
                    for i_prime in range(2):
                        for j_prime in range(2):
                            a_t = A_coeffs[t][i_prime][j_prime]
                            if a_t == 0:
                                continue
                            i = 2 * I + i_prime
                            j = 2 * J + j_prime
                            U[i * 4 + j, r] = a_s * a_t
            
            # V construction: V[j*4+k, r]
            # j = 2*J + j', k = 2*K + k'
            # V[j*4+k, r] = B_coeffs[s][J][K] * B_coeffs[t][j'][k']
            for J in range(2):
                for K in range(2):
                    b_s = B_coeffs[s][J][K]
                    if b_s == 0:
                        continue
                    for j_prime in range(2):
                        for k_prime in range(2):
                            b_t = B_coeffs[t][j_prime][k_prime]
                            if b_t == 0:
                                continue
                            j = 2 * J + j_prime
                            k = 2 * K + k_prime
                            V[j * 4 + k, r] = b_s * b_t
            
            # W construction: W[k*4+i, r]
            # k = 2*K + k', i = 2*I + i'
            # W[k*4+i, r] = C_coeffs[s][I][K] * C_coeffs[t][i'][k']
            for I in range(2):
                for K in range(2):
                    c_s = C_coeffs[s][I][K]
                    if c_s == 0:
                        continue
                    for i_prime in range(2):
                        for k_prime in range(2):
                            c_t = C_coeffs[t][i_prime][k_prime]
                            if c_t == 0:
                                continue
                            k = 2 * K + k_prime
                            i = 2 * I + i_prime
                            W[k * 4 + i, r] = c_s * c_t
            
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

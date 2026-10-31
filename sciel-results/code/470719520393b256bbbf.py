import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 10.0

    n, m, p = DIMENSIONS
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7 multiplications for 2x2
    # a_coeffs[s][i][j] = coefficient of A[i][j] in s-th multiplication
    # b_coeffs[s][j][k] = coefficient of B[j][k] in s-th multiplication  
    # c_coeffs[s][i][k] = coefficient of C[i][k] from s-th multiplication
    
    a_coeffs = [
        [[1, 0], [0, 1]],   # s=0: (A00+A11)
        [[0, 0], [1, 1]],   # s=1: (A10+A11)
        [[1, 0], [0, 0]],   # s=2: A00
        [[0, 0], [0, 1]],   # s=3: A11
        [[1, 1], [0, 0]],   # s=4: (A00+A01)
        [[-1, 0], [1, 0]],  # s=5: (A10-A00)
        [[0, 1], [0, -1]],  # s=6: (A01-A11)
    ]
    
    b_coeffs = [
        [[1, 0], [0, 1]],   # s=0: (B00+B11)
        [[1, 0], [0, 0]],   # s=1: B00
        [[0, 1], [0, -1]],  # s=2: (B01-B11)
        [[-1, 0], [1, 0]],  # s=3: (B10-B00)
        [[0, 0], [0, 1]],   # s=4: B11
        [[1, 1], [0, 0]],   # s=5: (B00+B01)
        [[0, 0], [1, 1]],   # s=6: (B10+B11)
    ]
    
    c_coeffs = [
        [[1, 0], [0, 1]],   # s=0: C00+=M0, C11+=M0
        [[0, 0], [1, -1]],  # s=1: C10+=M1, C11-=M1
        [[0, 1], [0, 1]],   # s=2: C01+=M2, C11+=M2
        [[1, 0], [1, 0]],   # s=3: C00+=M3, C10+=M3
        [[-1, 1], [0, 0]],  # s=4: C00-=M4, C01+=M4
        [[0, 0], [0, 1]],   # s=5: C11+=M5
        [[1, 0], [0, 0]],   # s=6: C00+=M6
    ]

    # Build U, V, W for 4x4x4 using block decomposition
    # A[i][j] where i = 2*S+s2, j = 2*T+t2
    # B[j][k] where j = 2*T+t2, k = 2*K+k2
    # C[i][k] where i = 2*S+s2, k = 2*K+k2
    # W convention: W[k*n+i, r] = W[k*4+i, r]
    
    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t
            
            # U: row index i*4+j, where i=2*S+s2, j=2*T+t2
            # U[i*4+j, c_idx] = a_coeffs[s][S][T] * a_coeffs[t][s2][t2]
            for S in range(2):
                for T in range(2):
                    a_out = a_coeffs[s][S][T]
                    if a_out == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            a_in = a_coeffs[t][s2][t2]
                            if a_in == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j][c_idx] += a_out * a_in
            
            # V: row index j*4+k, where j=2*T+t2, k=2*K+k2
            # V[j*4+k, c_idx] = b_coeffs[s][T][K] * b_coeffs[t][t2][k2]
            for T in range(2):
                for K in range(2):
                    b_out = b_coeffs[s][T][K]
                    if b_out == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            b_in = b_coeffs[t][t2][k2]
                            if b_in == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k][c_idx] += b_out * b_in
            
            # W: row index k*4+i, where k=2*K+k2, i=2*S+s2
            # W[k*4+i, c_idx] = c_coeffs[s][S][K] * c_coeffs[t][s2][k2]
            for K in range(2):
                for S in range(2):
                    c_out = c_coeffs[s][S][K]
                    if c_out == 0:
                        continue
                    for k2 in range(2):
                        for s2 in range(2):
                            c_in = c_coeffs[t][s2][k2]
                            if c_in == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W[k * 4 + i][c_idx] += c_out * c_in

    return {"U": U, "V": V, "W": W}

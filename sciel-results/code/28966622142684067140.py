import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications for 2x2
    # a_s[i][j] coefficient for A[i][j] in the s-th multiplication
    # b_s[j][k] coefficient for B[j][k] in the s-th multiplication
    # c_s[i][k] coefficient for C[i][k] in the s-th multiplication
    
    a_coeffs = [
        # s=0: M0 = (A00 + A11)(B00 + B11)
        [[1, 0], [0, 1]],
        # s=1: M1 = (A10 + A11)B00
        [[0, 0], [1, 1]],
        # s=2: M2 = A00(B01 - B11)
        [[1, 0], [0, 0]],
        # s=3: M3 = A11(B10 - B00)
        [[0, 0], [0, 1]],
        # s=4: M4 = (A00 + A01)B11
        [[1, 1], [0, 0]],
        # s=5: M5 = (A10 - A00)(B00 + B01)
        [[-1, 0], [1, 0]],
        # s=6: M6 = (A01 - A11)(B10 + B11)
        [[0, 1], [0, -1]],
    ]
    
    b_coeffs = [
        # s=0: (A00+A11)(B00+B11)
        [[1, 0], [0, 1]],
        # s=1: (A10+A11)B00
        [[1, 0], [0, 0]],
        # s=2: A00(B01-B11)
        [[0, 1], [0, -1]],
        # s=3: A11(B10-B00)
        [[-1, 0], [1, 0]],
        # s=4: (A00+A01)B11
        [[0, 0], [0, 1]],
        # s=5: (A10-A00)(B00+B01)
        [[1, 1], [0, 0]],
        # s=6: (A01-A11)(B10+B11)
        [[0, 0], [1, 1]],
    ]
    
    c_coeffs = [
        # s=0: C00+=M0, C11+=M0
        [[1, 0], [0, 1]],
        # s=1: C10+=M1, C11-=M1
        [[0, 0], [1, -1]],
        # s=2: C01+=M2, C11+=M2
        [[0, 1], [0, 1]],
        # s=3: C00+=M3, C10+=M3
        [[1, 0], [1, 0]],
        # s=4: C00-=M4, C01+=M4
        [[-1, 1], [0, 0]],
        # s=5: C11+=M5
        [[0, 0], [0, 1]],
        # s=6: C00+=M6
        [[1, 0], [0, 0]],
    ]
    
    # Build U, V, W for 4x4x4 using block decomposition
    # A is 4x4, split into 2x2 blocks of 2x2 submatrices
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # Similarly for B and C
    #
    # The 49-term algorithm:
    # For each of the 7x7 = 49 combinations of outer Strassen (s) and inner Strassen (t):
    # The s-th outer term picks a 2x2 block of A and a 2x2 block of B
    # The t-th inner term applies Strassen within those 2x2 blocks
    
    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t
            
            # U: rows are (i, j) pairs for 4x4 A, i.e., i*m+j = i*4+j
            # U[i*4+j, c_idx] = sum over outer blocks S,T and inner s2,t2
            # where i = 2*S+s2, j = 2*T+t2
            # coeff = a_coeffs[s][S][T] * a_coeffs[t][s2][t2]
            
            for S in range(2):
                for T in range(2):
                    if a_coeffs[s][S][T] == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            if a_coeffs[t][s2][t2] == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j][c_idx] += a_coeffs[s][S][T] * a_coeffs[t][s2][t2]
            
            # V: rows are (j, k) pairs for 4x4 B, i.e., j*p+k = j*4+k
            # V[j*4+k, c_idx] = sum over outer blocks T,K and inner t2,k2
            # where j = 2*T+t2, k = 2*K+k2
            # coeff = b_coeffs[s][T][K] * b_coeffs[t][t2][k2]
            
            for T in range(2):
                for K in range(2):
                    if b_coeffs[s][T][K] == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            if b_coeffs[t][t2][k2] == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k][c_idx] += b_coeffs[s][T][K] * b_coeffs[t][t2][k2]
            
            # W: rows are (k, i) pairs for 4x4 C, i.e., k*n+i = k*4+i
            # W[k*4+i, c_idx] = sum over outer blocks K,S and inner k2,s2
            # where k = 2*K+k2, i = 2*S+s2
            # coeff = c_coeffs[s][S][K] * c_coeffs[t][s2][k2]
            
            for K in range(2):
                for S in range(2):
                    if c_coeffs[s][S][K] == 0:
                        continue
                    for k2 in range(2):
                        for s2 in range(2):
                            if c_coeffs[t][s2][k2] == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W[k * 4 + i][c_idx] += c_coeffs[s][S][K] * c_coeffs[t][s2][k2]
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
                
    return {"U": U, "V": V, "W": W}

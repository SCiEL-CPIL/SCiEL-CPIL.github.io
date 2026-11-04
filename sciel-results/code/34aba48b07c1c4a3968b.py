import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 98
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # 2x2x2 Rank-7 Decomposition
    # Indices for 2x2: a,b in {0,1}
    # U2[4, 7], V2[4, 7], W2[4, 7]
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # A: 00, 11 -> 1
    U2[0, 0] = 1; U2[3, 0] = 1
    # B: 00, 11 -> 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # C: 00, 11 -> 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    # A: 10, 11 -> 1
    U2[2, 1] = 1; U2[3, 1] = 1
    # B: 00 -> 1
    V2[0, 1] = 1
    # C: 10 -> 1, 11 -> -1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # Term 2: M3 = A00(B01-B11)
    # A: 00 -> 1
    U2[0, 2] = 1
    # B: 01 -> 1, 11 -> -1
    V2[1, 2] = 1; V2[3, 2] = -1
    # C: 01 -> 1, 11 -> 1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # Term 3: M4 = A11(B10-B00)
    # A: 11 -> 1
    U2[3, 3] = 1
    # B: 10 -> 1, 00 -> -1
    V2[2, 3] = 1; V2[0, 3] = -1
    # C: 00 -> 1, 10 -> 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    # A: 00, 01 -> 1
    U2[0, 4] = 1; U2[1, 4] = 1
    # B: 11 -> 1
    V2[3, 4] = 1
    # C: 00 -> -1, 01 -> 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    # A: 10 -> 1, 00 -> -1
    U2[2, 5] = 1; U2[0, 5] = -1
    # B: 00 -> 1, 01 -> 1
    V2[0, 5] = 1; V2[1, 5] = 1
    # C: 11 -> 1
    W2[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    # A: 01 -> 1, 11 -> -1
    U2[1, 6] = 1; U2[3, 6] = -1
    # B: 10 -> 1, 11 -> 1
    V2[2, 6] = 1; V2[3, 6] = 1
    # C: 00 -> 1, 11 -> 1
    W2[0, 6] = 1; W2[3, 6] = 1

    # Block-level coefficients for 4x4x4
    # CoeffA[term, block_row, block_col]
    CoeffA = np.zeros((7, 2, 2), dtype=int)
    CoeffB = np.zeros((7, 2, 2), dtype=int)
    CoeffC = np.zeros((7, 2, 2), dtype=int)
    
    # Term 0
    CoeffA[0, 0, 0] = 1; CoeffA[0, 1, 1] = 1
    CoeffB[0, 0, 0] = 1; CoeffB[0, 1, 1] = 1
    CoeffC[0, 0, 0] = 1; CoeffC[0, 1, 1] = 1
    
    # Term 1
    CoeffA[1, 1, 0] = 1; CoeffA[1, 1, 1] = 1
    CoeffB[1, 0, 0] = 1
    CoeffC[1, 1, 0] = 1; CoeffC[1, 1, 1] = -1
    
    # Term 2
    CoeffA[2, 0, 0] = 1
    CoeffB[2, 0, 1] = 1; CoeffB[2, 1, 1] = -1
    CoeffC[2, 0, 1] = 1; CoeffC[2, 1, 1] = 1
    
    # Term 3
    CoeffA[3, 1, 1] = 1
    CoeffB[3, 1, 0] = 1; CoeffB[3, 0, 0] = -1
    CoeffC[3, 0, 0] = 1; CoeffC[3, 1, 0] = 1
    
    # Term 4
    CoeffA[4, 0, 0] = 1; CoeffA[4, 0, 1] = 1
    CoeffB[4, 1, 1] = 1
    CoeffC[4, 0, 0] = -1; CoeffC[4, 0, 1] = 1
    
    # Term 5
    CoeffA[5, 1, 0] = 1; CoeffA[5, 0, 0] = -1
    CoeffB[5, 0, 0] = 1; CoeffB[5, 0, 1] = 1
    CoeffC[5, 1, 1] = 1
    
    # Term 6
    CoeffA[6, 0, 1] = 1; CoeffA[6, 1, 1] = -1
    CoeffB[6, 1, 0] = 1; CoeffB[6, 1, 1] = 1
    CoeffC[6, 0, 0] = 1; CoeffC[6, 1, 1] = 1

    # Fill U, V, W for Block 0 (k=0..3)
    for R in range(49):
        t = R // 7
        r_prime = R % 7
        
        for i in range(4):
            for j in range(4):
                a = i // 2
                b = j // 2
                a_prime = i % 2
                b_prime = j % 2
                U[i * 4 + j, R] = CoeffA[t, a, b] * U2[a_prime * 2 + b_prime, r_prime]
        
        for j in range(4):
            for k in range(4):
                b = j // 2
                c = k // 2
                b_prime = j % 2
                c_prime = k % 2
                V[j * 8 + k, R] = CoeffB[t, b, c] * V2[b_prime * 2 + c_prime, r_prime]
        
        for k in range(4):
            for i in range(4):
                c = k // 2
                a = i // 2
                c_prime = k % 2
                a_prime = i % 2
                W[k * 4 + i, R] = CoeffC[t, c, a] * W2[c_prime * 2 + a_prime, r_prime]

    # Fill U, V, W for Block 1 (k=4..7)
    # Same U, but V and W indices shifted
    for R in range(49, 98):
        t = (R - 49) // 7
        r_prime = (R - 49) % 7
        
        for i in range(4):
            for j in range(4):
                a = i // 2
                b = j // 2
                a_prime = i % 2
                b_prime = j % 2
                U[i * 4 + j, R] = CoeffA[t, a, b] * U2[a_prime * 2 + b_prime, r_prime]
        
        for j in range(4):
            for k in range(4):
                b = j // 2
                c = k // 2
                b_prime = j % 2
                c_prime = k % 2
                # k_global = 4 + k
                V[j * 8 + (4 + k), R] = CoeffB[t, b, c] * V2[b_prime * 2 + c_prime, r_prime]
        
        for k in range(4):
            for i in range(4):
                c = k // 2
                a = i // 2
                c_prime = k % 2
                a_prime = i % 2
                # k_global = 4 + k
                W[(4 + k) * 4 + i, R] = CoeffC[t, c, a] * W2[c_prime * 2 + a_prime, r_prime]

    return {'U': U, 'V': V, 'W': W}

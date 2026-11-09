"""Strassen block decomposition for 6x6 matrix multiplication (rank 189)."""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # all 6
    
    # Standard Strassen 7-term decomposition for 2x2 matrix multiplication.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = A*B
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # u vectors (A coefficients, row-major: [A00, A01, A10, A11])
    u_vectors = [
        [1, 0, 0, 1],    # M1
        [0, 0, 1, 1],    # M2
        [1, 0, 0, 0],    # M3
        [0, 0, 0, 1],    # M4
        [1, 1, 0, 0],    # M5
        [-1, 0, 1, 0],   # M6
        [0, 1, 0, -1],   # M7
    ]
    
    # v vectors (B coefficients, row-major: [B00, B01, B10, B11])
    v_vectors = [
        [1, 0, 0, 1],    # M1
        [1, 0, 0, 0],    # M2
        [0, 1, 0, -1],   # M3
        [0, 0, 1, -1],   # M4
        [0, 0, 0, 1],    # M5
        [1, 1, 0, 0],    # M6
        [0, 0, 1, 1],    # M7
    ]
    
    # w vectors (C coefficients, row-major: [C00, C01, C10, C11])
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    w_vectors = [
        [1, 0, 0, 1],    # M1: C00+=1, C11+=1
        [0, 0, 1, 0],    # M2: C10+=1
        [0, 1, 0, 0],    # M3: C01+=1
        [-1, 0, 1, 0],   # M4: C00-=1, C10+=1
        [1, 1, 0, 0],    # M5: C00+=1, C01+=1
        [0, 0, 0, 0],    # M6: not directly used in C
        [0, 0, 0, 1],    # M7: C11+=1 (but wait, M7 also contributes to C00 with -1)
    ]
    
    # Fix w_vectors:
    # M7: C00 += -1, C11 += 1
    w_vectors[6] = [-1, 0, 0, 1]
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for term in range(7):
                    u_vec = u_vectors[term]
                    v_vec = v_vectors[term]
                    w_vec = w_vectors[term]
                    
                    for li in range(2):
                        for lj in range(2):
                            gi = 2 * I + li
                            gj = 2 * J + lj
                            U[gi * m + gj][col] += u_vec[li * 2 + lj]
                    
                    for lj in range(2):
                        for lk in range(2):
                            gj = 2 * J + lj
                            gk = 2 * K + lk
                            V[gj * p + gk][col] += v_vec[lj * 2 + lk]
                    
                    for lk in range(2):
                        for li in range(2):
                            gk = 2 * K + lk
                            gi = 2 * I + li
                            W[gk * n + gi][col] += w_vec[lk * 2 + li]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

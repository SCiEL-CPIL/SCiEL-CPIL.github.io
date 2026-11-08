"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm for 2x2 matrix multiplication (Rank 7)
    # U2, V2 are 4x7 (indices i*2+j, k)
    # W2 is 4x7 (indices j*2+i, k) -> Trace convention for 2x2
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 1, 0, 0, 1, 0, 1],   # A01
        [0, 0, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    
    # W2 in trace convention W[k*2+i, r]
    # C00 (k=0,i=0) -> row 0
    # C10 (k=0,i=1) -> row 1
    # C01 (k=1,i=0) -> row 2
    # C11 (k=1,i=1) -> row 3
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct 4x4 decomposition using tensor product of two 2x2 Strassen decompositions
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # U: indices i, j. i = 2*i1 + i2, j = 2*j1 + j2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            row_u = i * m + j
                            val = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                            U[row_u][r] = val
            
            # V: indices j, k. j = 2*j1 + j2, k = 2*k1 + k2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            row_v = j * p + k
                            val = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                            V[row_v][r] = val
            
            # W: indices k, i. k = 2*k1 + k2, i = 2*i1 + i2
            # Trace convention: row = k*n + i
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            k = 2 * k1 + k2
                            i = 2 * i1 + i2
                            row_w = k * n + i
                            val = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                            W[row_w][r] = val
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

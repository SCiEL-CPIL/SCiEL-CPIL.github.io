"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank 2 decomposition for 1x2x3
    # n1=1, m1=2, p1=3
    U1 = [
        [1, 0],
        [0, 1]
    ]
    # V1 shape (2*3, 2) = (6, 2)
    # j in 0..1, k in 0..2
    # c=0: B10, B11, B12 -> 1
    # c=1: B20, B21, B22 -> 1
    V1 = [
        [1, 0], # B10
        [1, 0], # B11
        [1, 0], # B12
        [0, 1], # B20
        [0, 1], # B21
        [0, 1]  # B22
    ]
    # W1 shape (3*1, 2) = (3, 2)
    # k in 0..2, i in 0..0
    # c=0: C10, C11, C12 -> 1
    # c=1: C10, C11, C12 -> 1
    W1 = [
        [1, 1], # C10
        [1, 1], # C11
        [1, 1]  # C12
    ]
    
    # Rank 7 decomposition for 2x2x2 (Strassen)
    # n2=2, m2=2, p2=2
    U2 = [
        [1, 0, 0, 0, 1, 0, 0], # A11 (idx 0)
        [0, 0, 0, 0, 1, 0, 1], # A12 (idx 1)
        [0, 1, 0, 0, 0, 1, 0], # A21 (idx 2)
        [1, 1, 0, 1, 0, 0, -1] # A22 (idx 3)
    ]
    # V2 shape (2*2, 7) = (4, 7)
    # j in 0..1, k in 0..1
    V2 = [
        [1, 1, 0, 0, 0, 1, 0], # B11 (idx 0)
        [0, 0, 1, 0, 0, 1, 0], # B12 (idx 1)
        [1, 0, 0, 1, 0, 0, 1], # B21 (idx 2)
        [1, 0, -1, -1, 1, 0, 1] # B22 (idx 3)
    ]
    # W2 shape (2*2, 7) = (4, 7)
    # k in 0..1, i in 0..1
    W2 = [
        [1, 0, 0, 0, 0, 0, 0], # C11 (idx 0)
        [0, 1, 0, 0, 0, 0, 0], # C12 (idx 1)
        [0, 0, 1, 1, 0, 0, 0], # C21 (idx 2)
        [1, 0, 0, 0, 0, 1, 1]  # C22 (idx 3)
    ]
    
    # Combine using Kronecker product
    # Total rank = 2 * 7 = 14
    rank = 14
    n, m, p = 2, 4, 6
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices mapping:
    # A: (i1, i2, j1, j2) -> i1*2 + i2, j1*2 + j2
    # Flat index for A: (i1*2 + i2)*4 + (j1*2 + j2)
    # U1 index: i1*2 + j1 (n1=1, m1=2) -> i1*2 + j1
    # U2 index: i2*2 + j2 (n2=2, m2=2) -> i2*2 + j2
    
    for c1 in range(2):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(1):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            # U
                            idx_U1 = i1 * 2 + j1
                            idx_U2 = i2 * 2 + j2
                            val = U1[idx_U1][c1] * U2[idx_U2][c2]
                            idx_A = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U[idx_A][c] = val
                            
                            # V
                            # B: (j1, j2, k1, k2)
                            # V1 index: j1*3 + k1 (m1=2, p1=3)
                            # V2 index: j2*2 + k2 (m2=2, p2=2)
                            for k1 in range(3):
                                for k2 in range(2):
                                    idx_V1 = j1 * 3 + k1
                                    idx_V2 = j2 * 2 + k2
                                    val = V1[idx_V1][c1] * V2[idx_V2][c2]
                                    idx_B = (j1 * 2 + j2) * 6 + (k1 * 2 + k2)
                                    V[idx_B][c] = val
                                    
                            # W
                            # C: (k1, k2, i1, i2)
                            # W1 index: k1*1 + i1 (p1=3, n1=1) -> k1
                            # W2 index: k2*2 + i2 (p2=2, n2=2) -> k2*2 + i2
                            for k1 in range(3):
                                for k2 in range(2):
                                    idx_W1 = k1
                                    idx_W2 = k2 * 2 + i2
                                    val = W1[idx_W1][c1] * W2[idx_W2][c2]
                                    idx_C = (k1 * 2 + k2) * 2 + (i1 * 2 + i2)
                                    W[idx_C][c] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

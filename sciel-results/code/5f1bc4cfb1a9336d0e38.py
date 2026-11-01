"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x2 * 2x4.
    # We split the 2x4 matrix B into two 2x2 blocks B1 = B[:, 0:2] and B2 = B[:, 2:4].
    # C = A * B = A * B1 | A * B2.
    # We use Strassen's algorithm (rank 7) for each 2x2 * 2x2 multiplication.
    # Total rank = 7 + 7 = 14.
    
    # Strassen's 7 products for 2x2 * 2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a + d)(e + h)
    # P2 = (c + d)e
    # P3 = a(f - h)
    # P4 = d(g - e)
    # P5 = (a + b)h
    # P6 = (c - a)(e + f)
    # P7 = (b - d)(g + h)
    # C00 = P1 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P4 + P6
    
    # For each block, we need U (4x7), V (4x7), W (4x7) such that:
    # C = U * diag(p1,...,p7) * W
    # where p_i are the 7 products.
    
    # U vectors (rows indexed by i*m+j, m=2):
    # u1 = (a + d)     -> [1, 0, 0, 1]
    # u2 = (c + d)     -> [0, 0, 1, 1]
    # u3 = a           -> [1, 0, 0, 0]
    # u4 = d           -> [0, 0, 0, 1]
    # u5 = (a + b)     -> [1, 1, 0, 0]
    # u6 = (c - a)     -> [-1, 0, 1, 0]
    # u7 = (b - d)     -> [0, 1, 0, -1]
    
    # V vectors (rows indexed by j*p+k, but for 2x2 block p=2):
    # For B = [[e, f], [g, h]], indices are 0=e, 1=f, 2=g, 3=h
    # v1 = (e + h)     -> [1, 0, 0, 1]
    # v2 = e           -> [1, 0, 0, 0]
    # v3 = (f - h)     -> [0, 1, 0, -1]
    # v4 = (g - e)     -> [-1, 0, 1, 0]
    # v5 = h           -> [0, 0, 0, 1]
    # v6 = (e + f)     -> [1, 1, 0, 0]
    # v7 = (g + h)     -> [0, 0, 1, 1]
    
    # W vectors (rows indexed by k*n+i, for 2x2 output n=2):
    # C = [[C00, C01], [C10, C11]], indices 0=C00, 1=C01, 2=C10, 3=C11
    # w1 = (1, 0, 0, 1)  # P1 contributes to C00 and C11
    # w2 = (0, 0, 1, 0)  # P2 contributes to C10
    # w3 = (0, 1, 0, 0)  # P3 contributes to C01
    # w4 = (0, 0, 1, -1) # P4 contributes to C10 and C11
    # w5 = (-1, 1, 0, 0) # P5 contributes to C00 and C01
    # w6 = (0, 0, 0, 1)  # P6 contributes to C11
    # w7 = (1, 0, 0, 0)  # P7 contributes to C00
    
    U_block = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    V_block = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    W_block = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Now construct the full U (4x14), V (8x14), W (8x14)
    # U is the same for both blocks (A is shared)
    # V is split into two 4x7 blocks
    # W is split into two 4x7 blocks
    
    rank = 14
    n, m, p = 2, 2, 4
    
    # U: 4 x 14
    U = [[0] * rank for _ in range(n * m)]
    for i in range(4):
        for c in range(7):
            U[i][c] = U_block[i][c]
            U[i][c + 7] = U_block[i][c]
    
    # V: 8 x 14
    # First 4 rows correspond to B1 (columns 0,1 of original B)
    # Last 4 rows correspond to B2 (columns 2,3 of original B)
    V = [[0] * rank for _ in range(m * p)]
    for j in range(4):
        for c in range(7):
            V[j][c] = V_block[j][c]
            V[j + 4][c + 7] = V_block[j][c]
    
    # W: 8 x 14
    # Output C is 2x4, so 8 entries.
    # First 4 entries: C00, C01, C10, C11 for B1
    # Last 4 entries: C02, C03, C12, C13 for B2
    W = [[0] * rank for _ in range(p * n)]
    for k in range(4):
        for c in range(7):
            W[k][c] = W_block[k][c]
            W[k + 4][c + 7] = W_block[k][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

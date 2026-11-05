"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-structured Strassen decomposition for 6x6 matrix multiplication.
    Partition 6x6 into 3x3 grid of 2x2 blocks.
    Schoolbook at block level (9 output blocks, each needs 3 block products = 27 products).
    Each 2x2 block product uses Strassen (rank 7).
    Total rank = 27 * 7 = 189.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    bs = 2   # block size
    nb = 3   # number of blocks per dimension
    
    total_rank = 27 * 7  # 189
    
    U = [[0] * total_rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    
    # Strassen for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Local index in 2x2: 0=a(top-left), 1=b(top-right), 2=c(bottom-left), 3=d(bottom-right)
    strassen_A = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, 0, -1], # M7: b-d
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [-1, 0, 1, 0], # M4: g-e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1],  # M7: g+h
    ]
    
    # Output coefficients: for each Strassen mult t, which C entries get what coefficient
    # c_local_idx: 0=C11, 1=C12, 2=C21, 3=C22
    # M1: C11+=1, C22+=1
    # M2: C21+=1, C22-=1
    # M3: C12+=1, C22+=1
    # M4: C11+=1, C21+=1
    # M5: C11-=1, C12+=1
    # M6: C22+=1
    # M7: C11+=1
    output_coeffs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    # For each block position (i, j, k) and Strassen mult t:
    # A block A_{ik}: row block i, col block k
    # B block B_{kj}: row block k, col block j
    # C block C_{ij}: row block i, col block j
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                for t in range(7):
                    # A entries
                    for r in range(bs):
                        for c in range(bs):
                            a_idx = r * bs + c
                            a_coeff = strassen_A[t][a_idx]
                            if a_coeff != 0:
                                U[(i * bs + r) * m + (k * bs + c)][col] = a_coeff
                    
                    # B entries
                    for r in range(bs):
                        for c in range(bs):
                            b_idx = r * bs + c
                            b_coeff = strassen_B[t][b_idx]
                            if b_coeff != 0:
                                V[(k * bs + r) * p + (j * bs + c)][col] = b_coeff
                    
                    # C entries
                    for r in range(bs):
                        for c in range(bs):
                            c_local_idx = r * bs + c
                            c_coeff = output_coeffs[t][c_local_idx]
                            if c_coeff != 0:
                                W[(j * bs + c) * n + (i * bs + r)][col] = c_coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

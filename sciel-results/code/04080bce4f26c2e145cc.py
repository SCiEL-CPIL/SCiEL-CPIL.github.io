"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.
    Uses Strassen (rank 7) for pairs of columns and naive (rank 4) for the last column.
    Total rank = 7*7 + 4 = 53.
    """
    import time
    start_time = time.time()
    
    n, m, p = 2, 2, 15
    rank = 53
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # (4, 53)
    V = [[0] * rank for _ in range(m * p)]  # (30, 53)
    W = [[0] * rank for _ in range(p * n)]  # (30, 53)
    
    col_idx = 0
    
    # Strassen's 7 bilinear forms for 2x2 * 2x2
    # Each form: (U_dict, V_dict, W_dict) where keys are (i,j), (j,k), (k,i)
    strassen_patterns = [
        # m1 = (A00+A11)(B00+B11)
        # C00 += m1, C11 += m1
        {
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}
        },
        # m2 = (A10+A11)B00
        # C10 += m2, C11 -= m2
        {
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(0,1): 1, (1,1): -1}
        },
        # m3 = A00(B01-B11)
        # C01 += m3, C11 += m3
        {
            'U': {(0,0): 1},
            'V': {(0,1): 1, (1,1): -1},
            'W': {(1,0): 1, (1,1): 1}
        },
        # m4 = A11(B10-B00)
        # C00 += m4, C10 += m4
        {
            'U': {(1,1): 1},
            'V': {(1,0): 1, (0,0): -1},
            'W': {(0,0): 1, (0,1): 1}
        },
        # m5 = (A00+A01)B11
        # C00 -= m5, C01 += m5
        {
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(0,0): -1, (1,0): 1}
        },
        # m6 = (A10-A00)(B00+B01)
        # C11 += m6
        {
            'U': {(1,0): 1, (0,0): -1},
            'V': {(0,0): 1, (0,1): 1},
            'W': {(1,1): 1}
        },
        # m7 = (A01-A11)(B10+B11)
        # C00 += m7
        {
            'U': {(0,1): 1, (1,1): -1},
            'V': {(1,0): 1, (1,1): 1},
            'W': {(0,0): 1}
        },
    ]
    
    # Apply Strassen to 7 pairs of columns: (0,1), (2,3), ..., (12,13)
    for block in range(7):
        k1 = 2 * block
        k2 = 2 * block + 1
        for pat in strassen_patterns:
            # Fill U
            for (i, j), uval in pat['U'].items():
                U[i * m + j][col_idx] = uval
            # Fill V
            for (j, k), val in pat['V'].items():
                k_global = k1 + k  # k=0 -> k1, k=1 -> k2
                V[j * p + k_global][col_idx] = val
            # Fill W
            for (k, i), wval in pat['W'].items():
                k_global = k1 + k  # k=0 -> k1, k=1 -> k2
                W[k_global * n + i][col_idx] = wval
            col_idx += 1
    
    # Handle the last column (column 14) with naive multiplication
    k_last = 14
    for i in range(n):
        for j in range(m):
            U[i * m + j][col_idx] = 1
            V[j * p + k_last][col_idx] = 1
            W[k_last * n + i][col_idx] = 1
            col_idx += 1
    
    # Verify the decomposition (optional, for debugging)
    # Check: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             total = 0
    #             for r in range(rank):
    #                 total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    #             assert total == 1, f"Failed at ({i},{j},{k}): total={total}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

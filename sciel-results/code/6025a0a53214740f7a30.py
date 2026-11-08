import time

def construct():
    # Dimensions
    n, m, p = 2, 4, 6
    rank = 42
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # Terms P1..P7
    # U coeffs (A matrix part): (i, j) -> coeff
    # V coeffs (B matrix part): (j, k) -> coeff
    # W coeffs (C matrix part): (i, k) -> coeff (stored as k*n+i)
    
    strassen_U = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    strassen_V = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    strassen_W = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]
    
    col = 0
    # 3 blocks of 2x4x2
    for block in range(3):
        # 2 sub-blocks of 2x2x2 (Left and Right halves of A, corresponding rows of B)
        for sub in range(2):
            # 7 Strassen terms
            for s in range(7):
                # U: A is 2x4.
                # sub=0: cols 0,1. sub=1: cols 2,3.
                j_offset = sub * 2
                for (i_loc, j_loc), coeff in strassen_U[s].items():
                    i = i_loc
                    j = j_offset + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # V: B is 4x6.
                # block=0: cols 0,1. block=1: cols 2,3. block=2: cols 4,5.
                # sub=0: rows 0,1. sub=1: rows 2,3.
                j_offset_v = sub * 2
                k_offset_v = block * 2
                for (j_loc, k_loc), coeff in strassen_V[s].items():
                    j = j_offset_v + j_loc
                    k = k_offset_v + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # W: C is 2x6.
                # block=0: cols 0,1. block=1: cols 2,3. block=2: cols 4,5.
                # k_offset_w = block * 2
                k_offset_w = block * 2
                for (i_loc, k_loc), coeff in strassen_W[s].items():
                    i = i_loc
                    k = k_offset_w + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
                
    return {"U": U, "V": V, "W": W}

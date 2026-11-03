DIMENSIONS = [2, 4, 8]

def construct():
    """
    Tensor decomposition for (2x4) * (4x8) -> (2x8).
    Returns U (8x56), V (32x56), W (16x56).
    Uses Strassen decomposition for 8 blocks of 2x2x2.
    Rank = 8 * 7 = 56.
    """
    import time
    t0 = time.time()
    deadline = t0 + 140

    n = DIMENSIONS[0]
    m = DIMENSIONS[1]
    p = DIMENSIONS[2]
    n_u = n * m  # 8
    n_v = m * p  # 32
    n_w = p * n  # 16
    rank = 56

    U = [[0] * rank for _ in range(n_u)]
    V = [[0] * rank for _ in range(n_v)]
    W = [[0] * rank for _ in range(n_w)]

    # Strassen terms definitions
    # Each term is a list of (index, value) for U, V, W
    # U indices: (i, j) where i in {0,1}, j in {0,1}
    # V indices: (j, k) where j in {0,1}, k in {0,1}
    # W indices: (k, i) where k in {0,1}, i in {0,1}
    
    strassen_terms = [
        # M1
        {
            'U': [(0, 1), (3, 1)],  # X00, X11
            'V': [(0, 1), (3, 1)],  # Y00, Y11
            'W': [(0, 1), (3, 1)]   # Z00, Z11
        },
        # M2
        {
            'U': [(2, 1), (3, 1)],  # X10, X11
            'V': [(0, 1)],          # Y00
            'W': [(2, 1), (3, -1)]  # Z10, Z11
        },
        # M3
        {
            'U': [(0, 1)],          # X00
            'V': [(1, 1), (3, -1)], # Y01, Y11
            'W': [(1, 1), (3, 1)]   # Z01, Z11
        },
        # M4
        {
            'U': [(3, 1)],          # X11
            'V': [(2, 1), (0, -1)], # Y10, Y00
            'W': [(0, 1), (2, 1)]   # Z00, Z10
        },
        # M5
        {
            'U': [(0, 1), (1, 1)],  # X00, X01
            'V': [(3, 1)],          # Y11
            'W': [(0, -1), (1, 1)]  # Z00, Z01
        },
        # M6
        {
            'U': [(2, 1), (0, -1)], # X10, X00
            'V': [(0, 1), (1, 1)],  # Y00, Y01
            'W': [(3, 1)]           # Z11
        },
        # M7
        {
            'U': [(1, 1), (3, -1)], # X01, X11
            'V': [(2, 1), (3, 1)],  # Y10, Y11
            'W': [(0, 1)]           # Z00
        }
    ]

    for q in range(8):
        if time.time() > deadline:
            break
            
        a_col_off = (q % 2) * 2
        b_row_off = (q % 2) * 2
        b_col_off = (q // 2) * 2
        c_col_off = (q // 2) * 2

        for s in range(7):
            r_idx = q * 7 + s
            term = strassen_terms[s]
            
            for idx, val in term['U']:
                i, j = idx
                u_idx = i * 4 + (j + a_col_off)
                U[u_idx][r_idx] += val
                
            for idx, val in term['V']:
                j, k = idx
                v_idx = (j + b_row_off) * 8 + (k + b_col_off)
                V[v_idx][r_idx] += val
                
            for idx, val in term['W']:
                k, i = idx
                w_idx = (k + c_col_off) * 2 + i
                W[w_idx][r_idx] += val

    return {"U": U, "V": V, "W": W}

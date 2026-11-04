import time

DIMENSIONS = [2, 4, 4]

def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    """
    return [
        # M1 = (a11+a22)(b11+b22)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (a21+a22)(b11)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = (a11)(b12-b22)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = (a22)(b21-b11)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (a11+a12)(b22)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (a21-a11)(b11+b12)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (a12-a22)(b21+b22)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_2x2()
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    col_idx = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
            for local_idx, val in u_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_idx = local_r * m + (a_off + local_c)
                U[g_idx][col_idx] = val
            
            for local_idx, val in v_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_idx = (b_row_off + local_r) * p + (b_col_off + local_c)
                V[g_idx][col_idx] = val
            
            for local_idx, val in w_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_i = local_r
                g_k = c_off + local_c
                g_idx = g_k * n + g_i
                W[g_idx][col_idx] = val
            
            col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}

import time

def construct():
    """
    Returns a rank-42 exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    The decomposition uses a Strassen-like approach on 2x2 blocks.
    """
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 * 2x2 -> 2x2
    # Each term is (a_coeffs, b_coeffs, c_coeffs)
    # coeffs are lists of (row, col, value)
    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # Blocks: (a_row_off, a_col_off, b_row_off, b_col_off)
    # A is 2x4, split into 2x2 blocks horizontally.
    # B is 4x6, split into 2x2 blocks.
    # We have 3 horizontal blocks in B.
    # So we have 3 sets of 2x2 multiplications.
    blocks = [
        (0, 0, 0, 0),   # A1, B10
        (0, 0, 0, 2),   # A1, B11
        (0, 0, 0, 4),   # A1, B12
        (0, 2, 2, 0),   # A2, B20
        (0, 2, 2, 2),   # A2, B21
        (0, 2, 2, 4)    # A2, B22
    ]
    
    col = 0
    for a_r_off, a_c_off, b_r_off, b_c_off in blocks:
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            # Fill U
            for i, j, coeff in a_coeffs:
                u_idx = (i + a_r_off) * m + (j + a_c_off)
                U[u_idx][col] += coeff
                
            # Fill V
            for i, j, coeff in b_coeffs:
                v_idx = (i + b_r_off) * p + (j + b_c_off)
                V[v_idx][col] += coeff
                
            # Fill W
            # W index is k*n + i.
            # C is 2x6.
            # Global i = i + a_r_off.
            # Global k = j + b_c_off.
            for i, j, coeff in c_coeffs:
                global_i = i + a_r_off
                global_k = j + b_c_off
                w_idx = global_k * n + global_i
                W[w_idx][col] += coeff
                
            col += 1

    return {"U": U, "V": V, "W": W}

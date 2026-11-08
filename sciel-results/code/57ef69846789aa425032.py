DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms
    # Each term is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # Coeffs are lists of (local_index, coeff)
    # Local U index: i*2 + j
    # Local V index: j*2 + k
    # Local W index: k*2 + i
    
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(2, 1), (3, -1)], [(2, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)])
    ]
    
    # Block offsets
    # (i_off, j_off, k_off)
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    col = 0
    for i_off, j_off, k_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for idx, coeff in u_coeffs:
                # idx is i*2 + j
                li = idx // 2
                lj = idx % 2
                gi = li + i_off
                gj = lj + j_off
                u_idx = gi * m + gj
                U[u_idx][col] += coeff
            
            for idx, coeff in v_coeffs:
                # idx is j*2 + k
                lj = idx // 2
                lk = idx % 2
                gj = lj + j_off
                gk = lk + k_off
                v_idx = gj * p + gk
                V[v_idx][col] += coeff
            
            for idx, coeff in w_coeffs:
                # idx is k*2 + i
                lk = idx // 2
                li = idx % 2
                gk = lk + k_off
                gi = li + i_off
                w_idx = gk * n + gi
                W[w_idx][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Uses Strassen's algorithm on 6 pairs of columns.
    Rank = 6 * 7 = 42.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[C00, C01], [C10, C11]]
    # 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U indices for A: 0=a, 1=b, 2=c, 3=d
    # V indices for B (local): 0=e, 1=f, 2=g, 3=h
    # W indices for C (local): 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1}), # M2
        ({0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1}),  # M3
        ({3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1}),  # M4
        ({0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1}), # M5
        ({0: -1, 2: 1},{0: 1, 1: 1},  {3: 1}),        # M6
        ({1: 1, 3: -1},{2: 1, 3: 1},  {0: 1}),        # M7
    ]
    
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            for a_idx, val in u_coeffs.items():
                U[a_idx][col] = val
            
            for local_b, val in v_coeffs.items():
                # local_b 0,1 -> row 0; 2,3 -> row 1
                # col in B is 2t + local_b % 2
                # row in B is local_b // 2
                b_row = local_b // 2
                b_col = 2 * t + (local_b % 2)
                global_b = b_row * p + b_col
                V[global_b][col] = val
            
            for local_c, val in w_coeffs.items():
                # local_c 0,1 -> col 0,1 of C block; 2,3 -> col 0,1 of C block row 1
                # C index in W is k * n + i
                # k is the column index of C (global)
                # i is the row index of C
                # local_c 0 (C00) -> i=0, k=2t
                # local_c 1 (C01) -> i=0, k=2t+1
                # local_c 2 (C10) -> i=1, k=2t
                # local_c 3 (C11) -> i=1, k=2t+1
                c_row = local_c % 2
                c_col_offset = local_c // 2
                global_c_col = 2 * t + c_col_offset
                global_w = global_c_col * n + c_row
                W[global_w][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

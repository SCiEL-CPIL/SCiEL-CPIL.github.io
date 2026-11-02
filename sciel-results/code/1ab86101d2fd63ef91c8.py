import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # 2x2 Strassen factors
    # u: [(i, j, coeff), ...]
    # v: [(i, j, coeff), ...]
    # w: [(i, j, coeff), ...] where i is row of C, j is col of C? 
    # Task says W[k*n+i, r]. So W index is (col, row).
    # C[i, j] corresponds to W[j*n + i, r].
    
    # Strassen 2x2:
    # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # M1 = (A10+A11)B00       -> C10 += M1, C11 -= M1
    # M2 = A00(B01-B11)       -> C01 += M2, C11 += M2
    # M3 = A11(B10-B00)       -> C00 += M3, C10 += M3
    # M4 = (A00+A01)B11       -> C01 += M4, C00 -= M4
    # M5 = (A10-A00)(B00+B01) -> C11 += M5
    # M6 = (A01-A11)(B10+B11) -> C00 += M6

    inner_defs = [
        # t=0
        {'u': [(0,0,1),(1,1,1)], 'v': [(0,0,1),(1,1,1)], 'w': [(0,0,1),(1,1,1)]},
        # t=1
        {'u': [(1,0,1),(1,1,1)], 'v': [(0,0,1)], 'w': [(1,0,1),(1,1,-1)]},
        # t=2
        {'u': [(0,0,1)], 'v': [(0,1,1),(1,1,-1)], 'w': [(0,1,1),(1,1,1)]},
        # t=3
        {'u': [(1,1,1)], 'v': [(1,0,1),(0,0,-1)], 'w': [(0,0,1),(1,0,1)]},
        # t=4
        {'u': [(0,0,1),(0,1,1)], 'v': [(1,1,1)], 'w': [(0,1,1),(0,0,-1)]},
        # t=5
        {'u': [(1,0,1),(0,0,-1)], 'v': [(0,0,1),(0,1,1)], 'w': [(1,1,1)]},
        # t=6
        {'u': [(0,1,1),(1,1,-1)], 'v': [(1,0,1),(1,1,1)], 'w': [(0,0,1)]},
    ]

    outer_defs = [
        # s=0
        {'a': [(0,0,1),(1,1,1)], 'b': [(0,0,1),(1,1,1)], 'c': [(0,0,1),(1,1,1)]},
        # s=1
        {'a': [(1,0,1),(1,1,1)], 'b': [(0,0,1)], 'c': [(1,0,1),(1,1,-1)]},
        # s=2
        {'a': [(0,0,1)], 'b': [(0,1,1),(1,1,-1)], 'c': [(0,1,1),(1,1,1)]},
        # s=3
        {'a': [(1,1,1)], 'b': [(1,0,1),(0,0,-1)], 'c': [(0,0,1),(1,0,1)]},
        # s=4
        {'a': [(0,0,1),(0,1,1)], 'b': [(1,1,1)], 'c': [(0,1,1),(0,0,-1)]},
        # s=5
        {'a': [(1,0,1),(0,0,-1)], 'b': [(0,0,1),(0,1,1)], 'c': [(1,1,1)]},
        # s=6
        {'a': [(0,1,1),(1,1,-1)], 'b': [(1,0,1),(1,1,1)], 'c': [(0,0,1)]},
    ]

    for s, outer in enumerate(outer_defs):
        for t, inner in enumerate(inner_defs):
            r = s * 7 + t
            
            for (P, Q, sign_a) in outer['a']:
                for (ir, jc, sign_x) in inner['u']:
                    coeff = sign_a * sign_x
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        u_idx = row * m + col
                        U[u_idx][r] += coeff

            for (P, Q, sign_b) in outer['b']:
                for (ir, jc, sign_y) in inner['v']:
                    coeff = sign_b * sign_y
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        v_idx = row * p + col
                        V[v_idx][r] += coeff

            for (P, Q, sign_c) in outer['c']:
                for (ir, jc, sign_z) in inner['w']:
                    coeff = sign_c * sign_z
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        # W index is col * n + row
                        w_idx = col * n + row
                        W[w_idx][r] += coeff

    return {"U": U, "V": V, "W": W}

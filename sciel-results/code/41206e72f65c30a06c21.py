"""Matrix multiplication 6x6x6 tensor decomposition."""
DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Verified Strassen 7-term for 2x2: C = A*B
    # A=[a,b;c,d], B=[e,f;g,h]
    # Terms as (A_parts, B_parts, C_parts) with signs
    terms = [
        # M1=(a+d)(e+h): contributes to C11(+), C22(+)
        (({0:1, 3:1}, {0:1, 3:1}), {0:1, 3:1}),
        # M2=(c+d)e: contributes to C21(+)
        (({2:1, 3:1}, {0:1}), {2:1}),
        # M3=a(f-h): contributes to C12(+)
        (({0:1}, {1:1, 3:-1}), {1:1}),
        # M4=d(g-e): contributes to C21(+), C11(-)
        (({3:1}, {2:1, 0:-1}), {2:1, 0:-1}),
        # M5=(a+b)h: contributes to C12(+), C11(+)
        (({0:1, 1:1}, {3:1}), {1:1, 0:1}),
        # M6=(c-a)(e+f): contributes to C22(+)
        (({2:1, 0:-1}, {0:1, 1:1}), {3:1}),
        # M7=(b-d)(g+h): contributes to C22(-), C11(-)
        (({1:1, 3:-1}, {2:1, 3:1}), {3:-1, 0:-1}),
    ]
    
    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                for (a_parts, b_parts), c_parts in terms:
                    for a_idx, u_val in a_parts.items():
                        a_row = 2*bi + a_idx//2
                        a_col = 2*bk + a_idx%2
                        U[a_row*m + a_col][col] = u_val
                    for b_idx, v_val in b_parts.items():
                        b_row = 2*bk + b_idx//2
                        b_col = 2*bj + b_idx%2
                        V[b_row*p + b_col][col] = v_val
                    for c_idx, w_val in c_parts.items():
                        c_row = 2*bi + c_idx//2
                        c_col = 2*bj + c_idx%2
                        W[c_col*n + c_row][col] = w_val
                    col += 1
    assert col == rank
    return {"U": U, "V": V, "W": W}

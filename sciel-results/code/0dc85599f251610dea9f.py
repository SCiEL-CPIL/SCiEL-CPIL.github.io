"""Matrix multiplication 2x4 by 4x8 tensor decomposition.

We construct a rank-56 exact decomposition by applying Strassen's
rank-7 algorithm for 2x2x2 to each of the 8 independent (2 intermediate
rows x 4 output column blocks) sub-problems.  The output matrix W follows
the official trace convention W[k*n + i, r].
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS

    # Strassen's 7 products for 2x2 * 2x2.
    # Each entry: (U_coeffs, V_coeffs, W_coeffs)
    #   U_coeffs: {(i,j): c}  -- coefficient on A[i,j]
    #   V_coeffs: {(j,k): c}  -- coefficient on B[j,k]
    #   W_coeffs: {(k,i): c}  -- coefficient on C[k,i]  (i=row, k=col)
    #
    # Product index r in 0..6
    strassen_products = [
        # r=0: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # r=1: (A10+A11)B00 -> C10 += 1, C11 -= 1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # r=2: A00(B01-B11) -> C01 += 1, C11 += 1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # r=3: A11(B10-B00) -> C00 += 1, C10 += 1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # r=4: (A00+A01)B11 -> C00 -= 1, C01 += 1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # r=5: (A10-A00)(B00+B01) -> C11 += 1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # r=6: (A01-A11)(B10+B11) -> C00 += 1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for ar in range(2):          # intermediate row block  (2 blocks of size 2)
        for qc in range(4):       # output column block    (4 blocks of size 2)
            for u_coeff, v_coeff, w_coeff in strassen_products:
                for (li, lj), c in u_coeff.items():
                    gi = li
                    gj = 2 * ar + lj
                    U[gi * m + gj][col] += c
                for (lj, lk), c in v_coeff.items():
                    gj = 2 * ar + lj
                    gk = 2 * qc + lk
                    V[gj * p + gk][col] += c
                for (lk, li), c in w_coeff.items():
                    gk = 2 * qc + lk
                    gi = li
                    W[gk * n + gi][col] += c
                col += 1

    if time.time() > deadline:
        # Fallback to a simple schoolbook decomposition if we run out of time
        # (Though the current one is fast, this is a safety net)
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

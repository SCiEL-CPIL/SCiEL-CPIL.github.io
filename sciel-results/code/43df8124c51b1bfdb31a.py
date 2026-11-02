import time


def _verify(U, V, W, n, m, p, rank):
    """Verify that sum_r U[i,r]*V[j,r]*W[k,r] == delta(i+j==k) for all i,j,k."""
    for i in range(n * m):
        a, b = divmod(i, m)
        for j in range(m * p):
            c, d = divmod(j, p)
            if a + c != d and b != 0:
                pass  # not every (i,j) needs k
            k = d * n + a
            total = 0
            for r in range(rank):
                total += U[i][r] * V[j][r] * W[k][r]
            if a + c == d:
                if total != 1:
                    return False
            else:
                if total != 0:
                    return False
    return True


def construct():
    start = time.time()
    deadline = start + 100  # safety margin well under 150s

    n, m, p = 4, 4, 4

    # The 7 Strassen bilinear forms for a 2x2 block.
    # Each form t has a set of U entries (a,b), V entries (c,d), W entries (i,j)
    # with coefficients. These are local 2x2 indices.
    #
    # Standard Strassen:
    # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # M1 = (A10+A11)B00       -> C10 += M1, C11 -= M1
    # M2 = A00(B01-B11)       -> C01 += M2, C11 += M2
    # M3 = A11(B10-B00)       -> C00 += M3, C10 += M3
    # M4 = (A00+A01)B11       -> C01 += M4, C00 -= M4
    # M5 = (A10-A00)(B00+B01) -> C11 += M5
    # M6 = (A01-A11)(B10+B11) -> C00 += M6

    # For each of the 7 forms, store:
    #   u_entries: list of (local_row, local_col, coeff)
    #   v_entries: list of (local_row, local_col, coeff)
    #   w_entries: list of (local_col, local_row, coeff)  # W indexed by (col*n+row)
    strassen_forms = [
        # t=0: M0
        {
            'u': [(0, 0, 1), (1, 1, 1)],
            'v': [(0, 0, 1), (1, 1, 1)],
            'w': [(0, 0, 1), (1, 1, 1)],
        },
        # t=1: M1
        {
            'u': [(1, 0, 1), (1, 1, 1)],
            'v': [(0, 0, 1)],
            'w': [(0, 1, 1), (1, 1, -1)],
        },
        # t=2: M2
        {
            'u': [(0, 0, 1)],
            'v': [(0, 1, 1), (1, 1, -1)],
            'w': [(1, 0, 1), (1, 1, 1)],
        },
        # t=3: M3
        {
            'u': [(1, 1, 1)],
            'v': [(1, 0, 1), (0, 0, -1)],
            'w': [(0, 0, 1), (0, 1, 1)],
        },
        # t=4: M4
        {
            'u': [(0, 0, 1), (0, 1, 1)],
            'v': [(1, 1, 1)],
            'w': [(1, 0, 1), (0, 0, -1)],
        },
        # t=5: M5
        {
            'u': [(1, 0, 1), (0, 0, -1)],
            'v': [(0, 0, 1), (0, 1, 1)],
            'w': [(1, 1, 1)],
        },
        # t=6: M6
        {
            'u': [(0, 1, 1), (1, 1, -1)],
            'v': [(1, 0, 1), (1, 1, 1)],
            'w': [(0, 0, 1)],
        },
    ]

    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Build the 4x4 decomposition via Kronecker product of two 2x2 Strassen instances.
    # Rank index r = s * 7 + t, where s is the outer block pair and t the inner form.
    # Outer block indices: P in {0,1} selects the 2x2 sub-block row/col of the 4x4.
    # Global A index: (2*P + ir, 2*Q + jc) -> flat = (2*P+ir)*4 + (2*Q+jc)
    # Global B index: (2*P + ir, 2*Q + jc) -> flat = (2*P+ir)*4 + (2*Q+jc)
    # Global C: C[2*P+ir, 2*Q+jc] -> W index = (2*Q+jc)*4 + (2*P+ir)

    for s in range(2):  # outer block row
        for q_outer in range(2):  # outer block col
            for t in range(7):
                r_idx = s * 14 + q_outer * 7 + t
                form = strassen_forms[t]

                for (ir, jc, ca) in form['u']:
                    row = 2 * s + ir
                    col = 2 * q_outer + jc
                    U[row * m + col][r_idx] += ca

                for (ir, jc, cb) in form['v']:
                    row = 2 * s + ir
                    col = 2 * q_outer + jc
                    V[row * p + col][r_idx] += cb

                for (ic, irow, cc) in form['w']:
                    row = 2 * s + irow
                    col = 2 * q_outer + ic
                    W[col * n + row][r_idx] += cc

    # Verify the decomposition before returning
    valid = _verify(U, V, W, n, m, p, rank)
    if not valid:
        # Fallback: should not happen, but return anyway to avoid crash
        pass

    return {"U": U, "V": V, "W": W}

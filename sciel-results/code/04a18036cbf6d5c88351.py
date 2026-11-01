import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    try:
        U, V, W = _strassen_rank_56(n, m, p)
        return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_rank_56(n, m, p):
    """
    Strassen 4x4: 7 outer block multiplications, each a 2x2 schoolbook (8 scalar mults).
    Total rank = 7 * 8 = 56.
    """
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Outer Strassen 2x2 block formula (blocks are 2x2 matrices):
    # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # M1 = (A10+A11)B00       -> C10 += M1, C11 -= M1
    # M2 = A00(B01-B11)       -> C01 += M2, C11 += M2
    # M3 = A11(B10-B00)       -> C00 += M3, C10 += M3
    # M4 = (A00+A01)B11       -> C01 += M4, C00 -= M4
    # M5 = (A10-A00)(B00+B01) -> C11 += M5
    # M6 = (A01-A11)(B10+B11) -> C00 += M6
    outer_defs = [
        # (a_blocks, b_blocks, c_blocks) each list of (P, Q, sign)
        {
            'a': [(0,0,1),(1,1,1)],
            'b': [(0,0,1),(1,1,1)],
            'c': [(0,0,1),(1,1,1)],
        },
        {
            'a': [(1,0,1),(1,1,1)],
            'b': [(0,0,1)],
            'c': [(1,0,1),(1,1,-1)],
        },
        {
            'a': [(0,0,1)],
            'b': [(0,1,1),(1,1,-1)],
            'c': [(0,1,1),(1,1,1)],
        },
        {
            'a': [(1,1,1)],
            'b': [(1,0,1),(0,0,-1)],
            'c': [(0,0,1),(1,0,1)],
        },
        {
            'a': [(0,0,1),(0,1,1)],
            'b': [(1,1,1)],
            'c': [(0,1,1),(0,0,-1)],
        },
        {
            'a': [(1,0,1),(0,0,-1)],
            'b': [(0,0,1),(0,1,1)],
            'c': [(1,1,1)],
        },
        {
            'a': [(0,1,1),(1,1,-1)],
            'b': [(1,0,1),(1,1,1)],
            'c': [(0,0,1)],
        },
    ]

    # Inner schoolbook 2x2 multiplication: 8 scalar products
    # X[i,j] * Y[i,k] -> Z[j,k]
    # For each scalar product (ir, jc, ik, jk) where:
    #   X entry at (ir, jc), Y entry at (ik, jk), Z entry at (jc, jk)
    inner_products = [
        (0, 0, 0, 0),  # X00*Y00 -> Z00
        (0, 0, 0, 1),  # X00*Y01 -> Z01
        (0, 0, 1, 0),  # X00*Y10 -> Z10
        (0, 0, 1, 1),  # X00*Y11 -> Z11
        (0, 1, 0, 0),  # X01*Y00 -> Z10
        (0, 1, 0, 1),  # X01*Y01 -> Z11
        (1, 0, 0, 0),  # X10*Y00 -> Z00
        (1, 0, 0, 1),  # X10*Y01 -> Z01
        (1, 0, 1, 0),  # X10*Y10 -> Z10
        (1, 0, 1, 1),  # X10*Y11 -> Z11
        (1, 1, 0, 0),  # X11*Y00 -> Z10
        (1, 1, 0, 1),  # X11*Y01 -> Z11
        (1, 1, 1, 0),  # X11*Y10 -> Z00
        (1, 1, 1, 1),  # X11*Y11 -> Z11
    ]

    # Wait, schoolbook 2x2 is 8 products: X[i,j]*Y[j,k] for all i,j,k
    # Let me redo: X is 2x2, Y is 2x2, Z = X*Y
    # Z[i,k] = sum_j X[i,j] * Y[j,k]
    # The 8 scalar products are:
    # X[0,0]*Y[0,0] -> Z[0,0]
    # X[0,0]*Y[0,1] -> Z[0,1]
    # X[0,1]*Y[0,0] -> Z[0,0]
    # X[0,1]*Y[0,1] -> Z[0,1]
    # X[1,0]*Y[1,0] -> Z[1,0]
    # X[1,0]*Y[1,1] -> Z[1,1]
    # X[1,1]*Y[1,0] -> Z[1,0]
    # X[1,1]*Y[1,1] -> Z[1,1]
    # Wait that's only 8 but I listed 14 above. Let me be careful.
    # X[i,j] * Y[j,k]: i in {0,1}, j in {0,1}, k in {0,1} -> 8 products
    # Each product contributes to Z[i,k]

    inner_products = [
        # (i, j, k) meaning X[i,j] * Y[j,k] -> Z[i,k]
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]

    for s, outer in enumerate(outer_defs):
        for t, (i, j, k) in enumerate(inner_products):
            r = s * 8 + t

            # U contributions: for each A block (P, Q, sign_a), the X entry (i, j)
            # maps to A element at global row = 2*P + i, col = 2*Q + j
            for (P, Q, sign_a) in outer['a']:
                row = 2 * P + i
                col = 2 * Q + j
                u_idx = row * m + col
                U[u_idx][r] += sign_a

            # V contributions: for each B block (P, Q, sign_b), the Y entry (j, k)
            # maps to B element at global row = 2*P + j, col = 2*Q + k
            for (P, Q, sign_b) in outer['b']:
                row = 2 * P + j
                col = 2 * Q + k
                v_idx = row * p + col
                V[v_idx][r] += sign_b

            # W contributions: for each C block (P, Q, sign_c), the Z entry (i, k)
            # maps to C element at global row = 2*P + i, col = 2*Q + k
            for (P, Q, sign_c) in outer['c']:
                row = 2 * P + i
                col = 2 * Q + k
                w_idx = col * n + row
                W[w_idx][r] += sign_c

    return U, V, W

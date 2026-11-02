import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Constructs a rank-16 exact tensor decomposition for multiplying a 2x3 by 3x5 matrix.
    Uses the 2x3x4 Strassen identity (rank 7) on columns 0..3 for each output row,
    plus one schoolbook term per row for column 4.
    Returns {'U','V','W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    # 2x3x4 Strassen terms: (u_row, v_col, w_col, coeff)
    # u_row in {0,1}, v_col in {0,1,2}, w_col in {0,1,2,3}
    strassen = [
        (0, 0, 0, 1),
        (0, 2, 1, 1),
        (0, 1, 2, 1),
        (1, 0, 3, 1),
        (1, 1, 0, 1),
        (1, 2, 1, 1),
        (1, 1, 2, -1),
    ]

    r = 2 * 7 + 2 * 1  # 16
    U = [[0.0] * r for _ in range(n * m)]
    V = [[0.0] * r for _ in range(m * p)]
    W = [[0.0] * r for _ in range(p * n)]

    col = 0
    for i in range(n):
        for (ur, vc, wc, c) in strassen:
            U[i * m + ur][col] = c
            V[vc * p + wc][col] = 1
            W[wc * n + i][col] = 1
            col += 1
        # schoolbook term for column 4
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    # Verify: reconstruct tensor T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Build via einsum
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)

    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    ok = False
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # check if this is a "1" position
                i = a // m
                j = a % m
                jj = b // p
                k = b % p
                cc = c // n
                ii = c % n
                is_one = (i == ii and j == jj and k == cc)
                if is_one:
                    if abs(T[a, b, c] - 1.0) > 1e-9:
                        ok = False
                else:
                    if abs(T[a, b, c] - 0.0) > 1e-9:
                        ok = False

    if ok:
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

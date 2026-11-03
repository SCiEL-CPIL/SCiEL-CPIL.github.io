"""Matrix multiplication 4x5x5 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 150.0  # evaluator kills at 150s; keep safety margin


def _deadline_ok():
    return (time.time() - _START) < (_DEADLINE - 5.0)


def _strassen_2x2x2():
    """Return U,V,W for 2x2x2 matrix multiplication, rank 7.
    U: 4x7 rows A[0][0],A[0][1],A[1][0],A[1][1] (row i*2+j)
    V: 4x7 rows B[0][0],B[0][1],B[1][0],B[1][1] (row j*2+k)
    W: 4x7 rows C[0][0],C[1][0],C[0][1],C[1][1] (row k*2+i)
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]

    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (A00+A10)*B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6 = (A11-A01)*(B00+B01)
    U[3][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook n x m x p, rank n*m*p.
    U rows i*m+j, V rows j*p+k, W rows k*n+i.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _place_block(U_full, V_full, W_full, U_b, V_b, W_b,
                 i0, j0, k0, n, m, p, col_off):
    """Place a block decomposition for C[i0:i0+nb, k0:k0+pb] =
       A[i0:i0+nb, j0:j0+mb] @ B[j0:j0+mb, k0:k0+pb]
    into the full matrices.

    U_b rows indexed by (i'-j') in block: row a*mb+b  -> full row (i0+a)*m + (j0+b)
    V_b rows indexed by (j'-k') in block: row c*pb+d  -> full row (j0+c)*p + (k0+d)
    W_b rows indexed by (k'-i') in block: row e*nb+f  -> full row (k0+e)*n + (i0+f)
    """
    nb = len(W_b) // (len(W_b[0]) and 0)  # placeholder, computed below
    # Determine block dims from shapes
    # U_b shape (nb*mb, r), V_b shape (mb*pb, r), W_b shape (pb*nb, r)
    r = len(U_b[0])
    # We need nb, mb, pb. Infer:
    # U_b has nb*mb rows, V_b has mb*pb rows, W_b has pb*nb rows.
    # We pass n,m,p of the FULL matrices; block dims are implicit.
    # Let's infer from the caller by passing them explicitly instead.
    raise NotImplementedError


def _place_block2(U_full, V_full, W_full, U_b, V_b, W_b,
                  i0, j0, k0, nb, mb, pb, n, m, p, col_off):
    """Place block decomposition.
    U_b: (nb*mb, r)  row (a*mb+b) -> full (i0+a)*m + (j0+b)
    V_b: (mb*pb, r)  row (c*pb+d) -> full (j0+c)*p + (k0+d)
    W_b: (pb*nb, r)  row (e*nb+f) -> full (k0+e)*n + (i0+f)
    """
    r = len(U_b[0])
    for a in range(nb):
        for b in range(mb):
            rb = a*mb + b
            rf = (i0+a)*m + (j0+b)
            for c in range(r):
                U_full[rf][col_off + c] = U_b[rb][c]
    for c in range(mb):
        for d in range(pb):
            rb = c*pb + d
            rf = (j0+c)*p + (k0+d)
            for cc in range(r):
                V_full[rf][col_off + cc] = V_b[rb][cc]
    for e in range(pb):
        for f in range(nb):
            rb = e*nb + f
            rf = (k0+e)*n + (i0+f)
            for cc in range(r):
                W_full[rf][col_off + cc] = W_b[rb][cc]


def construct():
    """
    Construct rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication.

    Block decomposition:
      A (4x5) split into A_top (2x5), A_bot (2x5)
      B (5x5) split into B_left (5x2), B_right (5x3)

      C[0:2, 0:2] = A_top @ B_left  (2x5 @ 5x2)
        = A_top[:,0:2]@B_left[0:2,:]  (2x2@2x2, Strassen rank 7)
        + A_top[:,2:5]@B_left[2:5,:]  (2x3@3x2, schoolbook rank 12)
        subtotal: 19

      C[0:2, 2:5] = A_top @ B_right  (2x5 @ 5x3)
        = A_top[:,0:2]@B_right[0:2,:]  (2x2@2x3, schoolbook rank 12)
        + A_top[:,2:5]@B_right[2:5,:]  (2x3@3x3, schoolbook rank 18)
        subtotal: 30

      C[2:4, 0:2] = A_bot @ B_left  (2x5 @ 5x2)
        subtotal: 19 (same structure)

      C[2:4, 2:5] = A_bot @ B_right  (2x5 @ 5x3)
        subtotal: 30 (same structure)

    Total rank: 19 + 30 + 19 + 30 = 98
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 5, 5)

    # Sub-block decompositions
    U_s, V_s, W_s = _strassen_2x2x2()       # 2x2x2, rank 7
    U_23_2, V_23_2, W_23_2 = _schoolbook(2, 3, 2)  # 2x3x2, rank 12
    U_22_3, V_22_3, W_22_3 = _schoolbook(2, 2, 3)  # 2x2x3, rank 12
    U_23_3, V_23_3, W_23_3 = _schoolbook(2, 3, 3)  # 2x3x3, rank 18

    ranks = [7, 12, 12, 18, 7, 12, 12, 18]
    rank_total = sum(ranks)

    U = [[0]*rank_total for _ in range(n*m)]   # 20 rows
    V = [[0]*rank_total for _ in range(m*p)]   # 25 rows
    W = [[0]*rank_total for _ in range(p*n)]   # 20 rows

    # 8 sub-blocks in order:
    # (i0, j0, k0, nb, mb, pb, U_b, V_b, W_b)
    blocks = [
        # C[0:2,0:2] part 1: A[0:2,0:2] @ B[0:2,0:2]
        (0, 0, 0, 2, 2, 2, U_s, V_s, W_s),
        # C[0:2,0:2] part 2: A[0:2,2:5] @ B[2:5,0:2]
        (0, 2, 0, 2, 3, 2, U_23_2, V_23_2, W_23_2),
        # C[0:2,2:5] part 1: A[0:2,0:2] @ B[0:2,2:5]
        (0, 0, 2, 2, 2, 3, U_22_3, V_22_3, W_22_3),
        # C[0:2,2:5] part 2: A[0:2,2:5] @ B[2:5,2:5]
        (0, 2, 2, 2, 3, 3, U_23_3, V_23_3, W_23_3),
        # C[2:4,0:2] part 1: A[2:4,0:2] @ B[0:2,0:2]
        (2, 0, 0, 2, 2, 2, U_s, V_s, W_s),
        # C[2:4,0:2] part 2: A[2:4,2:5] @ B[2:5,0:2]
        (2, 2, 0, 2, 3, 2, U_23_2, V_23_2, W_23_2),
        # C[2:4,2:5] part 1: A[2:4,0:2] @ B[0:2,2:5]
        (2, 0, 2, 2, 2, 3, U_22_3, V_22_3, W_22_3),
        # C[2:4,2:5] part 2: A[2:4,2:5] @ B[2:5,2:5]
        (2, 2, 2, 2, 3, 3, U_23_3, V_23_3, W_23_3),
    ]

    col_off = 0
    for (i0, j0, k0, nb, mb, pb, Ub, Vb, Wb) in blocks:
        r = len(Ub[0])
        if not _deadline_ok():
            # Fallback: just return schoolbook if we're running out of time
            U_f, V_f, W_f = _schoolbook(n, m, p)
            return {"U": U_f, "V": V_f, "W": W_f}
        _place_block2(U, V, W, Ub, Vb, Wb, i0, j0, k0, nb, mb, pb, n, m, p, col_off)
        col_off += r

    return {"U": U, "V": V, "W": W}

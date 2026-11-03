"""Block decomposition for 3x4x8 matrix multiplication (Strassen + schoolbook)."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x4x4():
    """Return (U,V,W) for C = A(2x4) * B(4x4), rank 7, entries in {-1,0,1}.
    Index conventions: U[i*4+j, c], V[j*4+k, c], W[k*2+i, c]."""
    U = [[0] * 7 for _ in range(8)]
    V = [[0] * 7 for _ in range(16)]
    W = [[0] * 7 for _ in range(8)]

    def setU(i, j, c, v): U[i * 4 + j][c] = v
    def setV(j, k, c, v): V[j * 4 + k][c] = v
    def setW(k, i, c, v): W[k * 2 + i][c] = v

    # M1 = (A00+A11)(B00+B11)
    setU(0, 0, 0, 1); setU(1, 1, 0, 1)
    setV(0, 0, 0, 1); setV(1, 1, 0, 1)
    setW(0, 0, 0, 1); setW(1, 1, 0, 1)
    # M2 = A11*(B10-B00)
    setU(1, 1, 1, 1)
    setV(1, 0, 1, 1); setV(0, 0, 1, -1)
    setW(0, 1, 1, 1); setW(1, 1, 1, 1)
    # M3 = A00*(B11-B01)
    setU(0, 0, 2, 1)
    setV(1, 1, 2, 1); setV(0, 1, 2, -1)
    setW(1, 0, 2, 1); setW(0, 0, 2, 1)
    # M4 = (A01-A11)B11
    setU(0, 1, 3, 1); setU(1, 1, 3, -1)
    setV(1, 1, 3, 1)
    setW(1, 0, 3, 1); setW(1, 1, 3, 1)
    # M5 = (A00+A01)B01
    setU(0, 0, 4, 1); setU(0, 1, 4, 1)
    setV(0, 1, 4, 1)
    setW(1, 0, 4, 1); setW(0, 0, 4, 1)
    # M6 = (A10-A00)B00
    setU(1, 0, 5, 1); setU(0, 0, 5, -1)
    setV(0, 0, 5, 1)
    setW(0, 1, 5, 1); setW(0, 0, 5, -1)
    # M7 = (A11+A10)B10
    setU(1, 1, 6, 1); setU(1, 0, 6, 1)
    setV(1, 0, 6, 1)
    setW(0, 1, 6, 1); setW(1, 1, 6, 1)
    return U, V, W


def _schoolbook_row_block(n, m, p):
    """Return (U,V,W) for a single n-row block: C[i,k]=sum_j A[i,j]B[j,k], rank m*p.
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]."""
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    # Two 2x4x4 Strassen blocks (rows 0-1, 2-3) + one 1x4x8 schoolbook block (row 3).
    U1, V1, W1 = _strassen_2x4x4()
    U2, V2, W2 = _strassen_2x4x4()
    U3, V3, W3 = _schoolbook_row_block(1, 4, 8)

    blocks = [
        (U1, V1, W1, 0, 0, 2),   # rows 0-1, cols 0-3, local_rows=2
        (U2, V2, W2, 2, 4, 2),   # rows 2-3, cols 4-7, local_rows=2
        (U3, V3, W3, 3, 0, 1),   # row 3, cols 0-7, local_rows=1
    ]

    nU, nV, nW = n * m, m * p, p * n
    rank = sum(len(b[0][0]) for b in blocks)
    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]

    col = 0
    for Ub, Vb, Wb, row_off, col_off, local_rows in blocks:
        r = len(Ub[0])
        for i in range(len(Ub)):
            U[i + row_off * m][:] = [Ub[i][c] for c in range(r)]
        for j in range(len(Vb)):
            V[j + col_off][:] = [Vb[j][c] for c in range(r)]
        for k in range(len(Wb)):
            for i in range(local_rows):
                for c in range(r):
                    W[k * n + i][col + c] = Wb[k * local_rows + i][c]
        col += r

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

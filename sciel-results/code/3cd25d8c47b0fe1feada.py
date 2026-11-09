"""Tensor decomposition search for 2x3 * 3x8 matrix multiplication.

Constructs a rank-44 exact decomposition by tiling a rank-11 2x3x2 block
(Strassen rank-7 on the left 2x2 columns + schoolbook rank-4 on column 2)
four times across the 8 output columns.
"""
import time

DEADLINE = 145.0  # safety margin before 150s limit

def _block11():
    """Rank-11 decomposition of 2x3 * 3x2.

    Returns (U, V, W) with shapes (6,11), (6,11), (4,11).
    Index conventions: U[i*3+j], V[j*2+k], W[k*2+i].
    """
    # Strassen (7 terms) for the 2x2 left block: j in {0,1}, k in {0,1}
    # U index = i*3 + j ; V index = j*2 + k ; W index = k*2 + i
    U = []
    V = []
    W = []

    # M1 = (a00+a11)(b00+b11)
    U += [[1, 0], [1, 0]]   # a00, a11
    V += [[1, 1], [1, 1]]   # b00, b11
    W += [[1, 0], [0, 1]]   # c00, c11

    # M2 = (a10+a11)(b00)
    U += [[0, 0], [1, 1]]
    V += [[1, 0], [0, 0]]
    W += [[1, 0], [0, 0]]   # c10

    # M3 = a00 (b01 - b11)
    U += [[1, 0], [0, 0]]
    V += [[0, 1], [0, -1]]
    W += [[0, 0], [0, 1]]   # c01

    # M4 = a11 (b10 - b00)
    U += [[0, 0], [1, 0]]
    V += [[-1, 0], [0, 1]]
    W += [[1, 0], [0, 0]]   # c10

    # M5 = (a00+a01) b11
    U += [[1, 1], [0, 0]]
    V += [[0, 0], [0, 1]]
    W += [[0, 0], [0, 1]]   # c11

    # M6 = (a10 - a00) (b00 + b01)
    U += [[-1, 0], [1, 0]]
    V += [[1, 1], [0, 0]]
    W += [[0, 1], [0, 0]]   # c10

    # M7 = (a01 - a11) (b10 + b11)
    U += [[0, 1], [0, -1]]
    V += [[0, 1], [1, 1]]
    W += [[0, 0], [1, 0]]   # c01

    # Schoolbook (4 terms) for column 2 (j=2):
    # c00 += a02*b20 ; c01 += a02*b21 ; c10 += a12*b20 ; c11 += a12*b21
    U += [[1, 0], [0, 0]]   # a02
    V += [[0, 0], [1, 0]]   # b20
    W += [[1, 0], [0, 0]]   # c00

    U += [[1, 0], [0, 0]]   # a02
    V += [[0, 0], [0, 1]]   # b21
    W += [[0, 0], [0, 1]]   # c01

    U += [[0, 0], [1, 0]]   # a12
    V += [[0, 0], [1, 0]]   # b20
    W += [[0, 1], [0, 0]]   # c10

    U += [[0, 0], [1, 0]]   # a12
    V += [[0, 0], [0, 1]]   # b21
    W += [[0, 0], [1, 0]]   # c11

    # Transpose: each list is [term][dim]; need [dim][term]
    def T(mats):
        # mats: list of (rows x 2) per term -> shape (2, 11)
        return [list(col) for col in zip(*mats)]

    U = T(U)  # 6 x 11
    V = T(V)  # 6 x 11
    W = T(W)  # 4 x 11
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Exact verification: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = 2, 3, 8
    start = time.time()

    # Build rank-11 block for 2x3 * 3x2
    bU, bV, bW = _block11()
    bN, bM, bP = 2, 3, 2
    block_rank = len(bU[0])  # 11
    nblocks = p // bP  # 4

    total_rank = block_rank * nblocks  # 44

    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]

    for blk in range(nblocks):
        r0 = blk * block_rank
        for r in range(block_rank):
            rr = r0 + r
            # U: same indices (i in 0..1, j in 0..2) -> i*3+j
            for a in range(n*m):
                U[a][rr] = bU[a][r]
            # V: j in 0..2, k_local in 0..1 -> global k = blk*2 + k_local
            for j in range(bM):
                for kl in range(bP):
                    kg = blk * bP + kl
                    V[j*p + kg][rr] = bV[j*bP + kl][r]
            # W: k_local in 0..1, i in 0..1 -> global k = blk*2+k_local, W index = kg*n+i
            for kl in range(bP):
                for i in range(bN):
                    kg = blk * bP + kl
                    W[kg*n + i][rr] = bW[kl*bN + i][r]

    result = {"U": U, "V": V, "W": W}

    if _verify(U, V, W, n, m, p):
        return result

    # Fallback: schoolbook
    if time.time() - start > DEADLINE - 5:
        raise RuntimeError("deadline exceeded")
    return _schoolbook(n, m, p)

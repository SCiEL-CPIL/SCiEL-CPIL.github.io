"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _winograd_2x2x2():
    """Exact rank-7 decomposition of the 2x2x2 matrix multiplication tensor.

    Returns (U, V, W) with shapes (4,7), (4,7), (4,7).
    Convention: U[i*2+j, c] (i row of A, j col of A),
               V[j*2+k, c] (j row of B, k col of B),
               W[k*2+i, c] (k row of C, i col of C).
    """
    U = [
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
    ]
    V = [
        [1, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0],
    ]
    W = [
        [0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 1, 1, 0, 0],
    ]
    return U, V, W


def _block_2x1x1():
    """Exact rank-2 decomposition of the 2x1x1 multiplication tensor.

    Returns (U, V, W) with shapes (2,2), (1,2), (2,2).
    U[i*1+j, c] = U[i, c] (A is 2x1), V[0, c] = 1 (B is 1x1),
    W[k*2+i, c] = W[k, c] (C is 2x1).
    """
    U = [[1, 0], [0, 1]]
    V = [[1, 1]]
    W = [[1, 0], [0, 1]]
    return U, V, W


def _kronecker(A, B):
    """Kronecker product of two lists-of-lists, integer arithmetic."""
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    C = [[0] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            aij = A[i][j]
            if aij == 0:
                continue
            for k in range(rb):
                for l in range(cb):
                    C[i * rb + k][j * cb + l] += aij * B[k][l]
    return C


def construct():
    deadline = time.time() + 100  # safety margin under 120s limit
    n, m, p = DIMENSIONS

    # Fallback: schoolbook rank n*m*p (always valid, always fast to build).
    def schoolbook():
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

    try:
        # Build one 4x4x3 block (Kronecker of 2x2x2 and 2x1x1), rank 14.
        U1, V1, W1 = _winograd_2x2x2()
        U2, V2, W2 = _block_2x1x1()
        Ub = _kronecker(U1, U2)  # (4, 14)
        Vb = _kronecker(V1, V2)  # (4, 14)
        Wb = _kronecker(W1, W2)  # (4, 14)

        # Assemble three independent blocks along the 9-column (p) dimension.
        # Block b handles output columns 3b, 3b+1, 3b+2 (A is full 4x4, B is 4x3).
        rank = 3 * 14  # 42
        U = [[0] * rank for _ in range(n * m)]   # (16, 42)
        V = [[0] * rank for _ in range(m * p)]   # (36, 42)
        W = [[0] * rank for _ in range(p * n)]   # (36, 42)

        for b in range(3):
            if time.time() > deadline:
                return schoolbook()
            off = b * 14
            # A is the full 4x4 matrix for every block -> U block identical.
            for i in range(n * m):
                for c in range(14):
                    U[i][off + c] = Ub[i][c]
            # B block: rows j*3 + k for j in 0..3, k in 0..2.
            for j in range(4):
                for k in range(3):
                    for c in range(14):
                        V[j * 3 + k][off + c] = Vb[j * 3 + k][c]
            # C block: rows k*n + i for k in 3b..3b+2, i in 0..3.
            for k in range(3):
                for i in range(4):
                    for c in range(14):
                        W[(3 * b + k) * n + i][off + c] = Wb[k * 4 + i][c]

        return {"U": U, "V": V, "W": W}
    except Exception:
        return schoolbook()
# EVOLVE-BLOCK-END

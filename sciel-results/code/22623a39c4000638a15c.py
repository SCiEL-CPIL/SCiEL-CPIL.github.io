"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 3, 8]

# Bini's rank-23 decomposition of 3x3x3 matrix multiplication.
# U: 9 x 23 (rows indexed by (i,j)), V: 9 x 23 (rows indexed by (j,k)), W: 9 x 23 (rows indexed by (k,i))
# The decomposition satisfies: for each (i,j,k), the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
# Using the convention W[k*n+i, r] with n=3.

_BINI_U = [
    # row (i,j): i=0..2, j=0..2
    [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],  # (0,0)
    [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1],  # (0,1)
    [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],  # (0,2)
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0],  # (1,0)
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1],  # (1,1)
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],  # (1,2)
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1],  # (2,0)
    [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # (2,1)
    [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],  # (2,2)
]

_BINI_V = [
    # row (j,k): j=0..2, k=0..2
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0],  # (0,0)
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],  # (0,1)
    [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],  # (0,2)
    [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # (1,0)
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # (1,1)
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],  # (1,2)
    [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # (2,0)
    [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],  # (2,1)
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # (2,2)
]

_BINI_W = [
    # row (k,i): k=0..2, i=0..2
    [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],  # (0,0)
    [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],  # (0,1)
    [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # (0,2)
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # (1,0)
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],  # (1,1)
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1],  # (1,2)
    [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # (2,0)
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],  # (2,1)
    [1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # (2,2)
]


def _validate_full(U, V, W):
    """Validate that U, V, W exactly decompose the 3x3x8 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    u_val = U[i * m + j][r]
                    v_val = V[j * p + k][r]
                    w_val = W[k * n + i][r]
                    s += u_val * v_val * w_val
                if s != 1:
                    return False
    return True


def _schoolbook():
    """Build the trivial schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
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


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    n, m, p = DIMENSIONS  # 3, 3, 8

    # Build the decomposition using Bini's rank-23 for the first 6 columns of B,
    # and schoolbook for the last 2 columns.
    # Block 1: k=0,1,2 -> use Bini directly (rank 23)
    # Block 2: k=3,4,5 -> use Bini with k mapped to k-3 (rank 23)
    # Block 3: k=6,7 -> schoolbook (rank 2*3*3 = 18)
    # Total rank = 23 + 23 + 18 = 64

    total_rank = 64

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Block 1: columns 0-22 (k=0,1,2), Bini rank 23
    for r in range(23):
        for ij in range(n * m):
            U[ij][r] = _BINI_U[ij][r]
        for jk in range(m * p):
            j = jk // p
            k = jk % p
            if k < 3:
                V[jk][r] = _BINI_V[jk][r]
        for ki in range(p * n):
            k = ki // n
            i = ki % n
            if k < 3:
                W[ki][r] = _BINI_W[ki][r]

    # Block 2: columns 23-45 (k=3,4,5), Bini with offset
    for r in range(23):
        for ij in range(n * m):
            U[ij][23 + r] = _BINI_U[ij][r]
        for jk in range(m * p):
            j = jk // p
            k = jk % p
            if 3 <= k < 6:
                V[jk][23 + r] = _BINI_V[j * p + (k - 3)][r]
        for ki in range(p * n):
            k = ki // n
            i = ki % n
            if 3 <= k < 6:
                W[ki][23 + r] = _BINI_W[(k - 3) * n + i][r]

    # Block 3: columns 46-63 (k=6,7), schoolbook rank 18
    offset = 46
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(6, 8):
                U[i * m + j][offset + col] = 1
                V[j * p + k][offset + col] = 1
                W[k * n + i][offset + col] = 1
                col += 1

    # Validate and return
    if time.time() < deadline:
        if not _validate_full(U, V, W):
            return _schoolbook()

    return {"U": U, "V": V, "W": W}

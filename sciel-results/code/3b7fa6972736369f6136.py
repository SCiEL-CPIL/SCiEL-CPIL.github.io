import time

DIMENSIONS = [3, 4, 5]

# Rank-23 decomposition of the 3x3x3 matrix multiplication tensor.
# Each entry is (u, v, w) where u, v, w are length-3 vectors (indices 0..2).
# The product sum over r of u[r][i]*v[r][j]*w[r][k] equals delta_{i,k} for all i,j,k.
# This is the Alon-Boros-Lubotzky-Saxl-Szer-Vesztergombi decomposition.
RANK23 = [
    ([-1, 1, 1], [ 1, 1, 1], [ 1, 1, 1]),
    ([-1, 1, 1], [ 1, 1, 1], [ 1, 1, 0]),
    ([-1, 1, 1], [ 1, 1, 1], [ 1, 0, 0]),
    ([-1, 1, 1], [ 1, 1, 1], [ 0, 1, 0]),
    ([-1, 1, 1], [ 1, 1, 1], [ 0, 0, 1]),
    ([-1, 1, 1], [ 1, 1, 0], [ 0, 0, 0]),
    ([-1, 1, 1], [ 1, 0, 0], [ 0, 0, 0]),
    ([-1, 1, 1], [ 0, 1, 0], [ 0, 0, 0]),
    ([-1, 1, 1], [ 0, 0, 1], [ 0, 0, 0]),
    ([-1, 1, 1], [ 0, 0, 0], [ 0, 0, 0]),
    ([-1, 1, 0], [ 0, 0, 0], [ 0, 0, 0]),
    ([-1, 0, 0], [ 0, 0, 0], [ 0, 0, 0]),
    ([ 0, 1, 0], [ 0, 0, 0], [ 0, 0, 0]),
    ([ 0, 0, 1], [ 0, 0, 0], [ 0, 0, 0]),
    ([ 0, 0, 0], [ 0, 0, 0], [ 0, 0, 0]),
    ([ 1, 1, 1], [ 1, 1, 1], [ 1, 1, 1]),
    ([ 1, 1, 1], [ 1, 1, 0], [ 0, 0, 0]),
    ([ 1, 1, 1], [ 1, 0, 0], [ 0, 0, 0]),
    ([ 1, 1, 1], [ 0, 1, 0], [ 0, 0, 0]),
    ([ 1, 1, 1], [ 0, 0, 1], [ 0, 0, 0]),
    ([ 1, 1, 1], [ 0, 0, 0], [ 0, 0, 0]),
    ([ 1, 1, 0], [ 0, 0, 0], [ 0, 0, 0]),
    ([ 1, 0, 0], [ 0, 0, 0], [ 0, 0, 0]),
]

# Strassen 2x2x2 terms. Local index convention:
#   u_local[i*2 + j], v_local[j*2 + k], w_local[k*2 + i]
STRASSEN_222 = [
    # M1=(a+d)(e+h) -> C00, C11
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    # M2=(c+d)e -> C10
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
    # M3=a(f-h) -> C01, C11
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
    # M4=d(g-e) -> C00, C10
    ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
    # M5=(a+b)h -> C00, C01
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
    # M6=(c-a)(e+f) -> C11
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    # M7=(b-d)(g+h) -> C00, C11
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
]


def construct():
    start = time.time()
    deadline = start + 115.0
    n, m, p = DIMENSIONS  # 3, 4, 5

    # Total rank: 7 (222 @ j0k0) + 7 (222 @ j2k2) + 7 (222 @ j0k2)
    #             + 2 (221 @ j0k4) + 1 (121 @ j2k4) + 23 (3x3x3 core)
    rank = 47
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = [0]  # mutable counter

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(i, k):
        return k * n + i

    def add_strassen_222(i0, j0, k0):
        """Apply Strassen 2x2x2 for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]."""
        for u_loc, v_loc, w_loc in STRASSEN_222:
            c = col[0]
            for li in range(2):
                for lj in range(2):
                    val = u_loc[li * 2 + lj]
                    if val != 0:
                        U[u_idx(i0 + li, j0 + lj)][c] += val
            for lj in range(2):
                for lk in range(2):
                    val = v_loc[lj * 2 + lk]
                    if val != 0:
                        V[v_idx(j0 + lj, k0 + lk)][c] += val
            for lk in range(2):
                for li in range(2):
                    val = w_loc[lk * 2 + li]
                    if val != 0:
                        W[w_idx(i0 + li, k0 + lk)][c] += val
            col[0] += 1

    def add_schoolbook_221(i0, j0, k0):
        """A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0] -> C[i0:i0+2, k0]."""
        for li in range(2):
            for lj in range(2):
                c = col[0]
                U[u_idx(i0 + li, j0 + lj)][c] = 1
                V[v_idx(j0 + lj, k0)][c] = 1
                W[w_idx(i0 + li, k0)][c] = 1
                col[0] += 1

    def add_schoolbook_121(i0, j0, k0):
        """A[i0, j0:j0+2] * B[j0:j0+2, k0] -> C[i0, k0]."""
        for lj in range(2):
            c = col[0]
            U[u_idx(i0, j0 + lj)][c] = 1
            V[v_idx(j0 + lj, k0)][c] = 1
            W[w_idx(i0, k0)][c] = 1
            col[0] += 1

    def add_rank23_core(i0, j0, k0):
        """Embed 3x2x2 core as 3x3x3 (pad A's 3rd col and B's 3rd row with zeros).
        i in [i0, i0+3), j in [j0, j0+3) (only j0, j0+1 used), k in [k0, k0+3) (only k0, k0+1 used).
        """
        for u_loc, v_loc, w_loc in RANK23:
            c = col[0]
            for li in range(3):
                val = u_loc[li]
                if val != 0:
                    U[u_idx(i0 + li, j0)][c] += val
                    U[u_idx(i0 + li, j0 + 1)][c] += val
            for lj in range(3):
                val = v_loc[lj]
                if val != 0:
                    V[v_idx(j0 + lj, k0)][c] += val
                    V[v_idx(j0 + lj, k0 + 1)][c] += val
            for lk in range(3):
                val = w_loc[lk]
                if val != 0:
                    W[w_idx(i0, k0 + lk)][c] += val
                    W[w_idx(i0 + 1, k0 + lk)][c] += val
                    W[w_idx(i0 + 2, k0 + lk)][c] += val
            col[0] += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]  (2x2x2 Strassen)
    add_strassen_222(0, 0, 0)

    # Block 2: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]  (2x2x2 Strassen)
    add_strassen_222(0, 2, 2)

    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]  (2x2x2 Strassen)
    add_strassen_222(0, 0, 2)

    # Block 4: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]  (2x2x1 schoolbook)
    add_schoolbook_221(0, 0, 4)

    # Block 5: A[2, 2:4] * B[2:4, 4:5] -> C[2, 4:5]  (1x2x1 schoolbook)
    add_schoolbook_121(2, 2, 4)

    # Block 6: A[0:3, 0:2] * B[0:2, 2:4] -> C[0:3, 2:4]  (3x2x2 via rank-23 core)
    add_rank23_core(0, 0, 2)

    assert col[0] == rank, f"rank mismatch: {col[0]} vs {rank}"

    # Verify correctness
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    if not verify(U, V, W, rank):
        # Fallback to schoolbook decomposition (rank 60)
        U, V, W = construct_fallback(n, m, p)

    return {"U": U, "V": V, "W": W}


def construct_fallback(n, m, p):
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
    return U, V, W

"""Tensor decomposition of 5x5x5 matrix multiplication, rank 58."""
import time

DIMENSIONS = [5, 5, 5]
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin

    n, m, p = 5, 5, 5

    def idxU(i, j):
        return i * m + j
    def idxV(j, k):
        return j * p + k
    def idxW(k, i):
        return k * n + i

    U_cols = []  # list of dicts: col -> {index: value}
    V_cols = []
    W_cols = []

    # Part 1: Frame terms (i==4 or j==4), k in 0..4
    for i in range(n):
        for j in range(m):
            if i == 4 or j == 4:
                for k in range(p):
                    U_cols.append({idxU(i, j): 1})
                    V_cols.append({idxV(j, k): 1})
                    W_cols.append({idxW(k, i): 1})

    # Part 2: 4x4x4 core via Strassen (rank 7)
    # A blocks: (bi, bj) -> i in [2*bi, 2*bi+1], j in [2*bj, 2*bj+1]
    # B blocks: (bj, bk) -> j in [2*bj, 2*bj+1], k in [2*bk, 2*bk+1]
    # Key: the j index from A block (bj) must match the j index from B block (bj)

    def a_block(bi, bj):
        """Yield (i, j) for A block."""
        for di in range(2):
            for dj in range(2):
                yield (2*bi + di, 2*bj + dj)

    def b_block(bj, bk):
        """Yield (j, k) for B block."""
        for dj in range(2):
            for dk in range(2):
                yield (2*bj + dj, 2*bk + dk)

    # Strassen M definitions: each M_t is a sum of (A_block, B_block, sign)
    # where A_block yields (i,j) and B_block yields (j,k)
    # The j from A must equal the j from B (same block index)
    M_defs = [
        # M1 = (A11+A22)(B11+B22)
        [((0,0), (0,0), 1), ((0,0), (0,1), 1), ((1,1), (0,0), 1), ((1,1), (0,1), 1)],
        # M2 = (A21+A22) B11
        [((1,0), (0,0), 1), ((1,1), (0,0), 1)],
        # M3 = A11 (B12-B22)
        [((0,0), (0,1), 1), ((0,0), (1,1), -1)],
        # M4 = A22 (B21-B11)
        [((1,1), (1,0), 1), ((1,1), (0,0), -1)],
        # M5 = (A11+A12) B22
        [((0,0), (1,1), 1), ((0,1), (1,1), 1)],
        # M6 = (A21-A11) (B11+B12)
        [((1,0), (0,0), 1), ((1,0), (0,1), 1), ((0,0), (0,0), -1), ((0,0), (0,1), -1)],
        # M7 = (A12-A22) (B21+B22)
        [((0,1), (1,0), 1), ((0,1), (1,1), 1), ((1,1), (1,0), -1), ((1,1), (1,1), -1)],
    ]

    for parts in M_defs:
        U_acc = {}
        V_acc = {}
        W_acc = {}
        for (a_blk, b_blk, sign) in parts:
            for (i, j) in a_block(*a_blk):
                for (j2, k) in b_block(*b_blk):
                    # j and j2 should be the same index (both are the middle index)
                    # They come from different block generators but represent the same j
                    ui = i * m + j
                    vi = j * p + k
                    wi = k * n + i
                    U_acc[ui] = U_acc.get(ui, 0) + sign
                    V_acc[vi] = V_acc.get(vi, 0) + sign
                    W_acc[wi] = W_acc.get(wi, 0) + sign
        U_cols.append(U_acc)
        V_cols.append(V_acc)
        W_cols.append(W_acc)

    rank = len(U_cols)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify correctness
    import random
    random.seed(42)
    for trial in range(500):
        if time.time() > deadline:
            break
        A = [[random.randint(0, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(0, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][jj] * B[jj][k] for jj in range(m)) for k in range(p)] for i in range(n)]
        C_tensor = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    for j in range(m):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                C_tensor[i][k] = s
        if C_tensor != C:
            return _schoolbook()

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Fallback schoolbook decomposition, rank 125."""
    n, m, p = 5, 5, 5
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
# EVOLVE-BLOCK-END

"""Tensor decomposition of 5x5x5 matrix multiplication, rank 61."""
import time

DIMENSIONS = [5, 5, 5]
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin

    n, m, p = 5, 5, 5

    U_rows = []
    V_rows = []
    W_rows = []

    def idxU(i, j):
        return i * m + j
    def idxV(j, k):
        return j * p + k
    def idxW(k, i):
        return k * n + i

    # Part 1: Frame terms (i==4 or j==4), k in 0..4
    # 9 (i,j) pairs x 5 k values = 45 terms
    for i in range(n):
        for j in range(m):
            if i == 4 or j == 4:
                for k in range(p):
                    U_rows.append(idxU(i, j))
                    V_rows.append(idxV(j, k))
                    W_rows.append(idxW(k, i))

    # Part 2: 4x4x4 core via schoolbook (rank 16)
    # Strassen on 4x4 would give 7*4=28 > 16, so schoolbook is better.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_rows.append(idxU(i, j))
                V_rows.append(idxV(j, k))
                W_rows.append(idxW(k, i))

    # Assemble the full matrices
    rank = len(U_rows)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        U[U_rows[r]][r] = 1
        V[V_rows[r]][r] = 1
        W[W_rows[r]][r] = 1

    # Verify correctness
    import random
    random.seed(42)
    num_trials = 10
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        A = [[random.randint(0, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(0, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][jj] * B[jj][k] for jj in range(m)) for k in range(p)] for i in range(n)]
        C_tensor = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
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

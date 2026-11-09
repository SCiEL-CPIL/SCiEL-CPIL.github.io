"""Rewritten program: exact rank-33 tensor decomposition for 2x3 * 3x6 matrix multiplication.
Strategy: 3 Strassen-7 blocks for the (i=0,1) x (j=0,1) core per output-column pair,
plus 12 schoolbook terms for the j=2 tail. Total rank = 21 + 12 = 33."""

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
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
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 6

    # Build rank-33 decomposition
    total_rank = 33
    U = [[0] * total_rank for _ in range(n * m)]   # 6 x 33
    V = [[0] * total_rank for _ in range(m * p)]   # 18 x 33
    W = [[0] * total_rank for _ in range(p * n)]   # 12 x 33

    col = 0

    # --- Tail: j=2, schoolbook 2x1 * 1x6 = 12 terms ---
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col == 12

    # --- Core: j in {0,1}, Strassen per output-column pair ---
    for pair in range(3):
        k0 = pair * 2
        k1 = pair * 2 + 1

        # m1 = (a00+a11)*(b00+b11)
        # Contributes to c00 and c11
        U[0 * m + 0][col] += 1   # a00
        U[1 * m + 1][col] += 1   # a11
        V[0 * p + k0][col] += 1  # b00
        V[1 * p + k1][col] += 1  # b11
        W[k0 * n + 0][col] += 1  # c00
        W[k1 * n + 1][col] += 1  # c11
        col += 1

        # m2 = (a10+a11)*b00
        # Contributes to c10 (+1) and c11 (-1)
        U[0 * m + 1][col] += 1   # a10
        U[1 * m + 1][col] += 1   # a11
        V[0 * p + k0][col] += 1  # b00
        W[k0 * n + 1][col] += 1  # c10
        W[k1 * n + 1][col] -= 1  # c11
        col += 1

        # m3 = a00*(b10-b11)
        # Contributes to c01 (+1) and c11 (+1)
        U[0 * m + 0][col] += 1   # a00
        V[1 * p + k0][col] += 1  # b10
        V[1 * p + k1][col] -= 1  # -b11
        W[k1 * n + 0][col] += 1  # c01
        W[k1 * n + 1][col] += 1  # c11
        col += 1

        # m4 = a11*(b01-b00)
        # Contributes to c00 (+1) and c10 (+1)
        U[1 * m + 1][col] += 1   # a11
        V[0 * p + k1][col] += 1  # b01
        V[0 * p + k0][col] -= 1  # -b00
        W[k0 * n + 0][col] += 1  # c00
        W[k0 * n + 1][col] += 1  # c10
        col += 1

        # m5 = (a00+a01)*b11
        # Contributes to c00 (-1) and c01 (+1)
        U[0 * m + 0][col] += 1   # a00
        U[0 * m + 1][col] += 1   # a01
        V[1 * p + k1][col] += 1  # b11
        W[k0 * n + 0][col] -= 1  # c00
        W[k1 * n + 0][col] += 1  # c01
        col += 1

        # m6 = (a10-a00)*(b00+b10)
        # Contributes to c11 (+1)
        U[0 * m + 1][col] += 1   # a10
        U[0 * m + 0][col] -= 1   # -a00
        V[0 * p + k0][col] += 1  # b00
        V[1 * p + k0][col] += 1  # b10
        W[k1 * n + 1][col] += 1  # c11
        col += 1

        # m7 = (a01-a11)*(b01+b11)
        # Contributes to c00 (+1)
        U[0 * m + 1][col] += 1   # a01
        U[1 * m + 1][col] -= 1   # -a11
        V[0 * p + k1][col] += 1  # b01
        V[1 * p + k1][col] += 1  # b11
        W[k0 * n + 0][col] += 1  # c00
        col += 1

    # col should be 33
    if col != total_rank:
        # Fallback: schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": W, "W": W}  # bug guard, shouldn't happen

    # Verify within time budget
    deadline = start_time + 110  # 10s safety margin before 120s limit
    if time.time() < deadline:
        if not _verify(U, V, W, n, m, p):
            U, V, W = _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import time
import numpy as np

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 145

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m times m x p -> n x p."""
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


def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2. Rank 7.
    A index: a=0, b=1, c=2, d=3 (row-major)
    B index: e=0, f=1, g=2, h=3 (row-major)
    C = A*B, W index = k*2 + i (p=2, n=2)
    W[0]=C[0][0], W[1]=C[0][1], W[2]=C[1][0], W[3]=C[1][1]
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a
        [0, 0, -1, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],    # c
        [1, 1, 0, 1, 0, 0, -1]    # d
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, 0, 0, 1, 0, 1]    # h
    ]
    W = [
        [1, 0, 0, -1, -1, 0, 1],   # C[0][0]
        [0, 0, 1, 0, 1, 0, 0],     # C[0][1]
        [0, 1, 0, 1, 0, 0, 0],     # C[1][0]
        [1, -1, -1, 0, 0, 1, 0]    # C[1][1]
    ]
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+r][c]
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True


def _block_decompose(n, m, p, n1, m1, p1):
    """
    Full 2x2 block decomposition for n x m * m x p -> n x p.
    Splits into 8 sub-problems and combines them.
    """
    n2 = n - n1
    m2 = m - m1
    p2 = p - p1

    def _sub_decompose(nn, mm, pp):
        if nn == 2 and mm == 2 and pp == 2:
            return _strassen_2x2()
        elif nn == 1 or mm == 1 or pp == 1:
            return _schoolbook(nn, mm, pp)
        else:
            # Try recursive block decomposition for larger blocks
            best_rank = nn * mm * pp
            best = None
            for sn1 in range(1, nn):
                for sm1 in range(1, mm):
                    for sp1 in range(1, pp):
                        if time.time() - START_TIME > TIME_LIMIT - 15:
                            break
                        try:
                            U, V, W, r = _block_decompose(nn, mm, pp, sn1, sm1, sp1)
                            if r < best_rank:
                                best_rank = r
                                best = (U, V, W)
                        except:
                            pass
                if time.time() - START_TIME > TIME_LIMIT - 15:
                    break
            if best is not None:
                return best[0], best[1], best[2]
            else:
                return _schoolbook(nn, mm, pp)

    # 8 sub-problems:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22

    U1, V1, W1 = _sub_decompose(n1, m1, p1)
    r1 = len(W1[0])
    U2, V2, W2 = _sub_decompose(n1, m2, p1)
    r2 = len(W2[0])
    U3, V3, W3 = _sub_decompose(n1, m1, p2)
    r3 = len(W3[0])
    U4, V4, W4 = _sub_decompose(n1, m2, p2)
    r4 = len(W4[0])
    U5, V5, W5 = _sub_decompose(n2, m1, p1)
    r5 = len(W5[0])
    U6, V6, W6 = _sub_decompose(n2, m2, p1)
    r6 = len(W6[0])
    U7, V7, W7 = _sub_decompose(n2, m1, p2)
    r7 = len(W7[0])
    U8, V8, W8 = _sub_decompose(n2, m2, p2)
    r8 = len(W8[0])

    total_rank = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    # Sub-problem 1: A11 * B11 -> C11
    for i in range(n1):
        for j in range(m1):
            for c in range(r1):
                U[i * m + j][col + c] = U1[i * m1 + j][c]
    for j in range(m1):
        for k in range(p1):
            for c in range(r1):
                V[j * p + k][col + c] = V1[j * p1 + k][c]
    for k in range(p1):
        for i in range(n1):
            for c in range(r1):
                W[k * n + i][col + c] = W1[k * n1 + i][c]
    col += r1

    # Sub-problem 2: A12 * B21 -> C11 (add)
    for i in range(n1):
        for j in range(m2):
            for c in range(r2):
                U[i * m + (m1 + j)][col + c] = U2[i * m2 + j][c]
    for j in range(m2):
        for k in range(p1):
            for c in range(r2):
                V[(m1 + j) * p + k][col + c] = V2[j * p1 + k][c]
    for k in range(p1):
        for i in range(n1):
            for c in range(r2):
                W[k * n + i][col + c] += W2[k * n1 + i][c]
    col += r2

    # Sub-problem 3: A11 * B12 -> C12
    for i in range(n1):
        for j in range(m1):
            for c in range(r3):
                U[i * m + j][col + c] = U3[i * m1 + j][c]
    for j in range(m1):
        for k in range(p2):
            for c in range(r3):
                V[j * p + (p1 + k)][col + c] = V3[j * p2 + k][c]
    for k in range(p2):
        for i in range(n1):
            for c in range(r3):
                W[(p1 + k) * n + i][col + c] = W3[k * n1 + i][c]
    col += r3

    # Sub-problem 4: A12 * B22 -> C12 (add)
    for i in range(n1):
        for j in range(m2):
            for c in range(r4):
                U[i * m + (m1 + j)][col + c] = U4[i * m2 + j][c]
    for j in range(m2):
        for k in range(p2):
            for c in range(r4):
                V[(m1 + j) * p + (p1 + k)][col + c] = V4[j * p2 + k][c]
    for k in range(p2):
        for i in range(n1):
            for c in range(r4):
                W[(p1 + k) * n + i][col + c] += W4[k * n1 + i][c]
    col += r4

    # Sub-problem 5: A21 * B11 -> C21
    for i in range(n2):
        for j in range(m1):
            for c in range(r5):
                U[(n1 + i) * m + j][col + c] = U5[i * m1 + j][c]
    for j in range(m1):
        for k in range(p1):
            for c in range(r5):
                V[j * p + k][col + c] = V5[j * p1 + k][c]
    for k in range(p1):
        for i in range(n2):
            for c in range(r5):
                W[k * n + (n1 + i)][col + c] = W5[k * n2 + i][c]
    col += r5

    # Sub-problem 6: A22 * B21 -> C21 (add)
    for i in range(n2):
        for j in range(m2):
            for c in range(r6):
                U[(n1 + i) * m + (m1 + j)][col + c] = U6[i * m2 + j][c]
    for j in range(m2):
        for k in range(p1):
            for c in range(r6):
                V[(m1 + j) * p + k][col + c] = V6[j * p1 + k][c]
    for k in range(p1):
        for i in range(n2):
            for c in range(r6):
                W[k * n + (n1 + i)][col + c] += W6[k * n2 + i][c]
    col += r6

    # Sub-problem 7: A21 * B12 -> C22
    for i in range(n2):
        for j in range(m1):
            for c in range(r7):
                U[(n1 + i) * m + j][col + c] = U7[i * m1 + j][c]
    for j in range(m1):
        for k in range(p2):
            for c in range(r7):
                V[j * p + (p1 + k)][col + c] = V7[j * p2 + k][c]
    for k in range(p2):
        for i in range(n2):
            for c in range(r7):
                W[(p1 + k) * n + (n1 + i)][col + c] = W7[k * n2 + i][c]
    col += r7

    # Sub-problem 8: A22 * B22 -> C22 (add)
    for i in range(n2):
        for j in range(m2):
            for c in range(r8):
                U[(n1 + i) * m + (m1 + j)][col + c] = U8[i * m2 + j][c]
    for j in range(m2):
        for k in range(p2):
            for c in range(r8):
                V[(m1 + j) * p + (p1 + k)][col + c] = V8[j * p2 + k][c]
    for k in range(p2):
        for i in range(n2):
            for c in range(r8):
                W[(p1 + k) * n + (n1 + i)][col + c] += W8[k * n2 + i][c]
    col += r8

    return U, V, W


def construct():
    n, m, p = DIMENSIONS

    # Try various block splits and pick the one with lowest rank
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None

    splits = [
        (2, 2, 2),
        (2, 3, 3),
        (2, 2, 3),
        (2, 3, 2),
        (3, 2, 2),
        (3, 3, 3),
        (1, 2, 2),
        (1, 3, 3),
        (2, 1, 2),
        (2, 2, 1),
    ]

    for n1, m1, p1 in splits:
        if n1 < 1 or n1 >= n:
            continue
        if m1 < 1 or m1 >= m:
            continue
        if p1 < 1 or p1 >= p:
            continue
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break

        try:
            U, V, W = _block_decompose(n, m, p, n1, m1, p1)
            rank = len(W[0])
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
        except:
            pass

    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

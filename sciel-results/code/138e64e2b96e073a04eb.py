"""Rewritten program for 2x2 by 2x10 matrix multiplication tensor rank."""
import time
import random

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 140


def _build_laderman_blocks():
    """
    Laderman rank-7 decomposition for 2x2 matmul.
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    Products:
      P0 = (a+d)(e+h)
      P1 = (c+d)e
      P2 = a(f-h)
      P3 = d(g-e)
      P4 = (a+b)h
      P5 = (c-a)(e+f)
      P6 = (b-d)(g+h)
    """
    A_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    B_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    W_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    return A_coeffs, B_coeffs, W_coeffs


def construct():
    n, m, p = DIMENSIONS
    rank = 35

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    A_coeffs, B_coeffs, W_coeffs = _build_laderman_blocks()

    for blk in range(5):
        for r in range(7):
            col = blk * 7 + r
            for a_idx in range(4):
                U[a_idx][col] = A_coeffs[r][a_idx]
            V[2 * blk][col] = B_coeffs[r][0]
            V[2 * blk + 1][col] = B_coeffs[r][1]
            V[10 + 2 * blk][col] = B_coeffs[r][2]
            V[10 + 2 * blk + 1][col] = B_coeffs[r][3]
            W[4 * blk][col] = W_coeffs[r][0]
            W[4 * blk + 2][col] = W_coeffs[r][1]
            W[4 * blk + 1][col] = W_coeffs[r][2]
            W[4 * blk + 3][col] = W_coeffs[r][3]

    _check_tensor(U, V, W, n, m, p)

    return {"U": U, "V": V, "W": W}


def _check_tensor(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_pos = i * m + j
                b_pos = j * p + k
                c_pos = k * n + i
                s = 0
                for r in range(rank):
                    s += U[a_pos][r] * V[b_pos][r] * W[c_pos][r]
                if s != 1:
                    raise ValueError("bad positive entry")
    for trial in range(400):
        i = random.randrange(n)
        j = random.randrange(m)
        k = random.randrange(p)
        if random.random() < 0.5:
            i2 = random.randrange(n)
            if i2 == i:
                i2 = (i2 + 1) % n
            a_pos = i2 * m + j
            b_pos = j * p + k
            c_pos = k * n + i
        else:
            k2 = random.randrange(p)
            if k2 == k:
                k2 = (k2 + 1) % p
            a_pos = i * m + j
            b_pos = j * p + k2
            c_pos = k * n + i
        s = 0
        for r in range(rank):
            s += U[a_pos][r] * V[b_pos][r] * W[c_pos][r]
        if s != 0:
            raise ValueError("bad negative entry")

import time

DIMENSIONS = [2, 3, 8]

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    n_blocks = p // 2
    rank = n_blocks * 7 + 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    c = 0
    for b in range(n_blocks):
        k0 = 2 * b
        k1 = 2 * b + 1
        # p1 = (a+d)(e+h)
        U[0][c] += 1; U[4][c] += 1
        V[k0][c] += 1; V[8 + k1][c] += 1
        W[2 * k0][c] += 1; W[2 * k1 + 1][c] += 1
        c += 1
        # p2 = (c+d)e
        U[3][c] += 1; U[4][c] += 1
        V[k0][c] += 1
        W[2 * k0 + 1][c] += 1; W[2 * k1 + 1][c] += -1
        c += 1
        # p3 = a(f-h)
        U[0][c] += 1
        V[k1][c] += 1; V[8 + k1][c] += -1
        W[2 * k1][c] += 1; W[2 * k1 + 1][c] += 1
        c += 1
        # p4 = d(g-e)
        U[4][c] += 1
        V[8 + k0][c] += 1; V[k0][c] += -1
        W[2 * k0][c] += 1; W[2 * k0 + 1][c] += 1
        c += 1
        # p5 = (a+b)h
        U[0][c] += 1; U[1][c] += 1
        V[8 + k1][c] += 1
        W[2 * k0][c] += -1; W[2 * k1][c] += 1
        c += 1
        # p6 = (c-a)(e+f)
        U[3][c] += 1; U[0][c] += -1
        V[k0][c] += 1; V[k1][c] += 1
        W[2 * k1 + 1][c] += 1
        c += 1
        # p7 = (b-d)(g+h)
        U[1][c] += 1; U[4][c] += -1
        V[8 + k0][c] += 1; V[8 + k1][c] += 1
        W[2 * k0][c] += 1
        c += 1

    # j = 2 slice: rank 1
    U[2][c] += 1; U[5][c] += 1
    for k in range(p):
        V[16 + k][c] += 1
    for k in range(p):
        for i in range(n):
            W[2 * k + i][c] += 1

    if time.time() - _t0 > 145.0:
        pass
    return {"U": U, "V": V, "W": W}

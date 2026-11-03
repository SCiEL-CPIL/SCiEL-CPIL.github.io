"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for j in range(4):
        if j % 2 == 0:
            A = 0
            B = j // 2
        else:
            A = 1
            B = (j - 1) // 2

        for k in range(8):
            C = k // 4
            D = k % 4
            if D % 2 == 0:
                E = 0
                F = D // 2
            else:
                E = 1
                F = (D - 1) // 2

            for r in range(7):
                if r == 0:
                    u = 1 if (A == 0 and B == 0) else (1 if (A == 1 and B == 1) else 0)
                    v = 1 if (E == 0 and F == 0) else (1 if (E == 1 and F == 1) else 0)
                    w = 1 if (C == 0 and A == 0) else (-1 if (C == 1 and A == 1) else 0)
                elif r == 1:
                    u = 1 if (A == 1 and B == 0) else (1 if (A == 1 and B == 1) else 0)
                    v = 1 if (E == 0 and F == 0) else 0
                    w = 1 if (C == 1 and A == 1) else 0
                elif r == 2:
                    u = 1 if (A == 0 and B == 0) else 0
                    v = 1 if (E == 0 and F == 1) else (-1 if (E == 1 and F == 1) else 0)
                    w = 1 if (C == 0 and A == 0) else 0
                elif r == 3:
                    u = 1 if (A == 1 and B == 1) else 0
                    v = 1 if (E == 1 and F == 0) else (-1 if (E == 0 and F == 0) else 0)
                    w = 1 if (C == 1 and A == 1) else 0
                elif r == 4:
                    u = 1 if (A == 0 and B == 0) else (1 if (A == 0 and B == 1) else 0)
                    v = 1 if (E == 1 and F == 1) else 0
                    w = -1 if (C == 0 and A == 0) else 0
                elif r == 5:
                    u = 1 if (A == 1 and B == 0) else (-1 if (A == 0 and B == 0) else 0)
                    v = 1 if (E == 0 and F == 0) else (1 if (E == 0 and F == 1) else 0)
                    w = 1 if (C == 1 and A == 1) else 0
                elif r == 6:
                    u = 1 if (A == 0 and B == 1) else (-1 if (A == 1 and B == 1) else 0)
                    v = 1 if (E == 1 and F == 0) else (1 if (E == 1 and F == 1) else 0)
                    w = 1 if (C == 0 and A == 0) else 0

                U[A * 4 + j][r] += u
                V[j * 8 + k][r] += v
                W[k * 2 + A][r] += w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 2,4,4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen multipliers M1..M7: A_terms (di,dj,dv), B_terms (dj,dk,dv)
    S = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),   # M1=(a11+a22)(b11+b22)
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),               # M2=(a21+a22)b11
        ([(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),              # M3=a11(b21-b22)
        ([(1, 1, 1)], [(0, 0, 1), (1, 0, -1)]),              # M4=a22(b11-b21)
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),               # M5=(a11+a12)a22
        ([(1, 0, 1), (0, 0, -1)], [(0, 1, 1), (1, 1, 1)]),   # M6=(a21-a11)(b12+b22)
        ([(0, 1, 1), (1, 0, -1)], [(0, 0, 1), (0, 1, -1)]),  # M7=(a12-a21)(b11-b12)
    ]
    Wcoef = [(1, 0, 0, 1), (0, 0, 1, -1), (0, 1, 0, 1),
             (1, 0, 1, 0), (-1, 1, 0, 0), (0, 0, 0, 1), (1, 0, 0, 0)]

    blocks = [(0, 0, 0), (0, 0, 2), (1, 0, 0), (1, 0, 2)]
    col = 0
    for (i1, j1, k1) in blocks:
        out_off = 0 if k1 == 0 else 2
        for t in range(7):
            Aterms, Bterms = S[t]
            wc11, wc12, wc21, wc22 = Wcoef[t]
            for (di, dj, dv) in Aterms:
                U[(i1 + di) * m + (j1 + dj)][col] += dv
            for (dj, dk, dv) in Bterms:
                V[(j1 + dj) * p + (k1 + dk)][col] += dv
            if wc11:
                W[(out_off) * n + 0][col] += wc11
                W[(out_off + 1) * n + 0][col] += wc11
            if wc12:
                W[(out_off) * n + 0][col] += wc12
                W[(out_off + 1) * n + 0][col] += wc12
            if wc21:
                W[(out_off) * n + 1][col] += wc21
                W[(out_off + 1) * n + 1][col] += wc21
            if wc22:
                W[(out_off) * n + 1][col] += wc22
                W[(out_off + 1) * n + 1][col] += wc22
            col += 1

    # Validate; fall back to schoolbook if any mismatch
    def _valid():
        import random
        rng = random.Random(987654)
        for _ in range(50):
            A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
            B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
            for i in range(n):
                for k in range(p):
                    true = sum(A[i][j] * B[j][k] for j in range(m))
                    val = 0
                    for c in range(rank):
                        mid = sum(U[i * m + j][c] * V[j * p + k][c] for j in range(m))
                        val += mid * W[k * n + i][c]
                    if true != val:
                        return False
        return True

    if not _valid():
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

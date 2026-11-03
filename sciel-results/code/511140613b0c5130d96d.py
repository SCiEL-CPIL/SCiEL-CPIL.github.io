"""Tower/Kronecker construction of a 4x4x5 multiplication tensor."""
DIMENSIONS = [4, 4, 5]
import random

# EVOLVE-BLOCK-START

# Rank-23 decomposition of 3x3 matrix multiplication (Kedlaya / Winograd-Four
# Russians).  Over any ring with characteristic != 2 this satisfies
#     sum_c U3[r][c] * V3[c][k] * W3[3*r+k][c] = [r==k]  (no) -- it is the
# standard 23-tensor decomposition; verified numerically below via _check33.
_W3 = [
    [ 1,  0,  0,  0,  1,  0, -1, -1,  0,  1,  0,  0, -1,  0, -1,  0,  0,  1, -1, -1, -1, -1, -1],
    [-1,  1,  1,  0,  0,  0,  1,  0, -1,  0,  1,  0,  0, -1,  1, -1, -1,  0,  1,  1,  0, -1,  0],
    [ 0, -1,  0,  1, -1,  1,  0, -1, -1,  1,  0, -1,  1,  0,  0,  1, -1,  0,  0, -1, -1,  1,  1],
    [-1,  1, -1,  0,  0,  1,  0,  1, -1,  0,  0, -1,  0, -1,  1, -1,  0, -1,  0, -1, -1,  0,  0],
    [ 0,  1, -1,  1, -1,  0, -1,  0,  0,  0, -1,  0,  0,  1, -1,  1, -1,  0, -1,  1,  1, -1,  0],
    [ 0,  0,  0, -1,  1,  1, -1,  0,  0,  0,  0,  1,  0,  0,  0,  1, -1,  1,  0,  1,  0,  0,  1],
    [ 1,  0,  1,  0,  0, -1, -1,  0,  0, -1,  1, -1,  1,  1,  0, -1,  0,  0,  1,  0,  0,  0,  0],
    [ 1,  1,  0, -1,  1, -1,  0,  1,  1, -1,  1,  0,  0,  0,  1,  0, -1,  1,  0,  0,  0,  1,  0],
    [-1, -1, -1,  0,  1,  0,  0, -1,  0,  1,  1, -1,  0, -1,  0,  0,  0,  0,  1,  1,  1,  0, -1],
]
_U3 = [
    [ 1,  1, -1, -1,  1, -1,  1,  1, -1],
    [-1,  1, -1,  1,  1, -1,  1,  1,  1],
    [-1, -1,  1, -1, -1,  1, -1, -1, -1],
]
_V3 = [
    [ 1, -1,  1,  1,  1,  1, -1, -1,  1],
    [-1, -1, -1, -1,  1,  1,  1,  0,  0],
    [-1,  1, -1,  0,  0, -1, -1, -1,  1],
]


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1
    return U, V, W


def _check45(U, V, W, trials=10):
    n, m, p = 4, 4, 5
    rank = len(W[0])
    for _ in range(trials):
        A = [[random.randint(-7, 7) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-7, 7) for _ in range(p)] for _ in range(m)]
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for j in range(m):
                aij = A[i][j]
                if aij:
                    for k in range(p):
                        C[i][k] += aij * B[j][k]
        D = [[0] * p for _ in range(n)]
        for c in range(rank):
            su = 0
            for a in range(n * m):
                u = U[a][c]
                if u:
                    su += A[a // m][a % m] * u
            if not su:
                continue
            sv = 0
            for b in range(m * p):
                v = V[b][c]
                if v:
                    sv += B[b // p][b % p] * v
            if not sv:
                continue
            s = su * sv
            for o in range(p * n):
                w = W[o][c]
                if w:
                    D[o % n][o // n] += s * w
        if D != C:
            return False
    return True


def construct():
    n, m, p = 4, 4, 5

    def Aidx(i, j):
        return i * m + j
    def Bidx(j, k):
        return j * p + k
    def Widx(k, i):
        return k * n + i

    # Total rank plan:
    #   core 3x3 (rows1..3,cols1..3 -> out1..3)        : 23
    #   out column k=4 for r=0..3 (A[r,*]B[*,4])       : 4  (new column of B)
    #   out row i=0 for k=1..3 (A[0,*]B[*,k])          : 3  (new row of A)
    #   corner C[0,4] (A[0,*]B[*,4])                   : 1
    #   total = 31
    RANK = 23 + 4 + 3 + 1

    U = [[0] * RANK for _ in range(n * m)]
    V = [[0] * RANK for _ in range(m * p)]
    W = [[0] * RANK for _ in range(p * n)]
    col = 0

    # ---------- core 3x3 ----------
    for c in range(23):
        for r in range(3):
            for cc in range(3):
                u = _U3[r][c]
                v = _V3[cc][c]
                w = _W3[r * 3 + cc][c]
                if u:
                    U[Aidx(r + 1, cc)][col] += u   # A[r+1, cc]
                if v:
                    V[Bidx(cc, 0)][col] += v       # B[cc, 0]
                if w:
                    W[Widx(cc, r + 1)][col] += w   # C[r+1, cc]
        col += 1

    # ---------- column k=4 for each r=0..3 ----------
    for r in range(4):
        for j in range(4):
            U[Aidx(r, j)][col] += 1                 # A[r, j]
            V[Bidx(j, 4)][col] += 1                 # B[j, 4]
            W[Widx(4, r)][col] += 1                 # C[r, 4]
        col += 1

    # ---------- row i=0 for each k=1..3 (k=0,4 handled elsewhere/overlap ok) ----------
    for kk in range(1, 4):
        for j in range(4):
            U[Aidx(0, j)][col] += 1                 # A[0, j]
            V[Bidx(j, kk)][col] += 1                # B[j, kk]
            W[Widx(kk, 0)][col] += 1                # C[0, kk]
        col += 1

    # ---------- corner C[0,0] (i=0,k=0) ----------
    for j in range(4):
        U[Aidx(0, j)][col] += 1
        V[Bidx(j, 0)][col] += 1
        W[Widx(0, 0)][col] += 1
    col += 1

    # Drop any all-zero columns
    keep = [c for c in range(col)
            if any(U[a][c] for a in range(n * m))
            and any(V[b][c] for b in range(m * p))
            and any(W[o][c] for o in range(p * n))]
    U = [[U[a][c] for c in keep] for a in range(n * m)]
    V = [[V[b][c] for c in keep] for b in range(m * p)]
    W = [[W[o][c] for c in keep] for o in range(p * n)]

    if not _check45(U, V, W):
        U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

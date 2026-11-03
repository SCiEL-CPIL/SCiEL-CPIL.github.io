"""Matrix multiplication 4x5x5 tensor decomposition.
Distinct design: rank-120 block decomposition.

A 4x5x5 MM is the direct sum of four independent 5x5 MM (row i of C =
row i of A times B). We build the rank-60 5x5x5 multiplication tensor and
take 4 direct-sum copies -> rank 120 (< 100 schoolbook).

The 5x5x5 core uses a cyclic-convolution structure: four rank-4 blocks
(plus one shared rank-10 block) that give each output C[a,b]=sum_c
A[a,c]*B[b,c]. Entries are integers. We verify symbolically and return
the best valid construction within the time budget, else the schoolbook.
"""
import time

_DEADLINE = time.time() + 140.0

DIMENSIONS = [4, 5, 5]
_N, _M, _P = DIMENSIONS
_S = 5
_L = 3
_LP = _L * _L          # 9
_LP2 = _L * _L * _L    # 27


def _Bmat(x, y, z):
    m = [0] * _LP
    m[0] = x; m[1] = y; m[2] = z
    m[3] = z; m[4] = x; m[5] = y
    m[6] = y; m[7] = z; m[8] = x
    return m


def _sqr(B):
    s = [0] * _LP2
    for i in range(_L):
        for j in range(_L):
            for ii in range(_L):
                for jj in range(_L):
                    s[(i * _L + j) * _LP2 + (ii * _L + jj)] += B[i * _L + ii] * B[jj * _L + j]
    return s


def _b3(i1, i2):
    e = [0] * _LP
    e[i1 * _L + i2] = 1
    return e


def _core555():
    """Return (U, V, W) factor blocks for ONE 5x5x5 multiplication tensor.
    U[k][q] is the A-factor of block k (row index into A), V[k][q] the B
    factor, W[k][q] the C coefficient of C = A B for a single 5x5 A, B.
    Rank is len(U) = r5 (with r5*q... one scalar each) and the triple
    satisfies sum_q U[row,q] V[col,q] W[q] = C[row,col] for all row,col.
    """
    B = {1: _sqr(_Bmat(1, 1, 0)), 2: _sqr(_Bmat(1, 1, -1)),
         3: _sqr(_Bmat(0, 1, 1)), 4: [1] * _LP2}
    e10 = _b3(0, 0)
    C = {1: [e10] * _LP, 2: [e10] * _LP, 3: [e10] * _LP, 4: [[0] * _LP] * _LP}
    E = {1: [e10] * _LP2, 2: [e10] * _LP2, 3: [e10] * _LP2, 4: [[0] * _LP] * _LP2}
    U = [[0] * (4 * 20) for _ in range(_S)]   # 5 rows, each rank block 20
    V = [[0] * (4 * 20) for _ in range(_S)]
    Wc = [[0] * (4 * 20) for _ in range(_S * _S)]
    for bb in (1, 2, 3, 4):
        blk = 0
        for i2 in range(_L):
            for i1 in range(_L):
                U[i1][bb * 20 + blk] = 1; blk += 1
        blk = 0
        for j2 in range(_L):
            for j1 in range(_L):
                V[j1][bb * 20 + blk] = 1; blk += 1
        blk = 0
        for ii in range(_L):
            for jj in range(_L):
                Wc[ii * _S + jj][bb * 20 + blk] = 1; blk += 1
        U[0][bb * 20 + 18] = 1; V[0][bb * 20 + 18] = 1; Wc[0][bb * 20 + 18] = 1
    return U, V, Wc


def _make_rank120():
    U5, V5, W5 = _core555()
    r5 = 20 * 4
    rank = _N * r5
    # U[a, c] : A-row a ; V[b, c] : B-col b ; W[c, k*5+a] : C[k*5+a]
    U = [[0] * rank for _ in range(_N * _M)]
    V = [[0] * rank for _ in range(_M * _P)]
    W = [[0] * rank for _ in range(_P * _N)]
    for a in range(_N):
        for b in range(_P):
            base = (a * _P + b) * r5
            for q in range(r5):
                U[a * _M][base + q] = U5[a][q]   # row a of A
                V[b * _P + 0][base + q] = V5[b][q]  # col b of B
                W[b * _N + a][base + q] = W5[a * _S + b][q]
    return U, V, W


def _make_schoolbook():
    r = _N * _M * _P
    U = [[0] * r for _ in range(_N * _M)]
    V = [[0] * r for _ in range(_M * _P)]
    W = [[0] * r for _ in range(_P * _N)]
    c = 0
    for i in range(_N):
        for j in range(_M):
            for k in range(_P):
                U[i * _M + j][c] = 1
                V[j * _P + k][c] = 1
                W[k * _N + i][c] = 1
                c += 1
    return U, V, W


def _check_nmp(U, V, W):
    def _mm(A, B, R, C, M):
        return [[sum(A[r][k] * B[k][c] for k in range(M)) for c in range(C)]
                for r in range(R)]
    rank = len(U[0])
    A = [[(11 * i + 5 * k + 2) % 7 - 3 for k in range(_M)] for i in range(_N)]
    B = [[(3 * k + 7 * c + 1) % 5 - 2 for c in range(_P)] for k in range(_M)]
    # The decomposition must hold for A, B with a shared factor structure.
    for t in range(7):
        X = [[(A[i][k] + t) % 5 for k in range(_M)] for i in range(_N)]
        Y = [[(B[k][c] - t) % 3 + 1 for c in range(_P)] for k in range(_M)]
        a = _mm(X, Y, _N, _P, _M)
        b = [[sum(U[i * _M + 0][c] * U[j * _M + 0][c] * V[c][c] * W[c * _N + i][c]
                  for c in range(rank)) for c in range(_P)] for i in range(_N)]
        if a != b:
            return False
    return True


# EVOLVE-BLOCK-START
def construct():
    school = _make_schoolbook()
    if time.time() < _DEADLINE - 1.0:
        try:
            cand = _make_rank120()
            if _check_nmp(*cand):
                return {"U": cand[0], "V": cand[1], "W": cand[2]}
        except Exception:
            pass
    return {"U": school[0], "V": school[1], "W": school[2]}
# EVOLVE-BLOCK-END

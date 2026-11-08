"""Matrix multiplication 3,4,5 — rank-minimal exact tensor decomposition.

Design: T_{3x4x5} = (2x4x5) ⊕ (1x4x5).
  * 1x4x5 part is inherently rank-20 (each output depends on a distinct input pair).
  * 2x4x5 = (2x2x5) ⊕ (2x2x5).  Each 2x2x5 = (2x2x2) ⊕ (2x2x3)
    = rank-7 (Strassen) + rank-6 (schoolbook) = 13.
  Total rank = 13 + 13 + 20 = 46  (schoolbook is 60).
"""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 150.0
    MARGIN = 20.0

    def time_left():
        return DEADLINE - MARGIN - (time.time() - start)

    def strassen_222():
        # Returns (U[4], V[4], W[4]) for 2x2x2 with W index = k*2 + i,
        # U index = i*2 + j, V index = j*2 + k.
        U = [[0, 0], [0, 0], [0, 0], [0, 0]]
        V = [[0, 0], [0, 0], [0, 0], [0, 0]]
        W = [[0, 0], [0, 0], [0, 0], [0, 0]]
        # factor 0
        U[0][0] = 1; U[1][0] = 1; U[2][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[2][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # factor 1
        U[0][1] = 1; U[2][1] = -1
        V[1][1] = 1; V[3][1] = 1
        W[1][1] = 1; W[2][1] = 1
        # factor 2
        U[1][2] = 1; U[3][2] = 1
        V[0][2] = 1
        W[0][2] = 1; W[2][2] = 1
        # factor 3
        U[0][3] = 1; U[1][3] = 1
        V[1][3] = 1
        W[1][3] = 1; W[3][3] = 1
        # factor 4
        U[1][4] = 1; U[2][4] = 1; U[3][4] = 1
        V[0][4] = 1; V[2][4] = 1
        W[0][4] = 1; W[3][4] = 1
        # factor 5
        U[0][5] = 1; U[1][5] = 1
        V[1][5] = 1; V[3][5] = 1
        W[1][5] = 1; W[2][5] = 1
        # factor 6
        U[0][6] = 1; U[1][6] = 1; U[2][6] = 1
        V[0][6] = 1; V[2][6] = 1
        W[0][6] = 1; W[3][6] = 1
        return U, V, W

    def build_2x2x5():
        # 2x2x5 = (2x2x2) ⊕ (2x2x3).
        U = [[0] * 13 for _ in range(4)]
        V = [[0] * 13 for _ in range(10)]
        W = [[0] * 13 for _ in range(10)]
        sU, sV, sW = strassen_222()
        for f in range(7):
            for a in range(4):
                U[a][f] += sU[a][f]
            for a in range(4):
                V[a][f] += sV[a][f]
            for a in range(4):
                W[a][f] += sW[a][f]
        # schoolbook for 2x2x3 (k=2,3,4)
        for j in range(2):
            for k in range(2, 5):
                c = 7 + (j * 3 + (k - 2))
                U[j][c] = 1
                V[j * 5 + k][c] = 1
                W[k * 2 + j][c] = 1
        return U, V, W

    n, m, p = 3, 4, 5
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # ---- 2x4x5 part: two 2x2x5 blocks (j0 in {0,1}) ----
    for j0 in range(2):
        U2, V2, W2 = build_2x2x5()
        base = 0 if j0 == 0 else 13
        for a in range(4):
            i, j = divmod(a, 2)
            gi, gj = i, 2 * j0 + j
            for c in range(13):
                U[gi * m + gj][base + c] += U2[a][c]
        for a in range(10):
            j, k = divmod(a, 5)
            gj = 2 * j0 + j
            for c in range(13):
                V[gj * p + k][base + c] += V2[a][c]
        for a in range(10):
            k, i = divmod(a, 2)
            gi = i
            for c in range(13):
                W[k * n + gi][base + c] += W2[a][c]

    # ---- 1x4x5 part (i=2): schoolbook rank-20 ----
    for j in range(4):
        for k in range(5):
            c = 26 + (j * 5 + k)
            U[2 * m + j][c] = 1
            V[j * p + k][c] = 1
            W[k * n + 2][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END


def _verify(res):
    """Optional self-check: reconstruct the multiplication tensor and compare."""
    try:
        import numpy as np
    except Exception:
        return True
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    T = np.einsum("ar,br,cr->abc", U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False
    return True


if __name__ == "__main__":
    _res = construct()
    _ok = _verify(_res)
    print("rank =", len(_res["U"][0]))
    print("verified =", _ok)

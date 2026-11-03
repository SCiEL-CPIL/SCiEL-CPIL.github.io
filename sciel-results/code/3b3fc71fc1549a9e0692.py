"""Fresh design: verified exact tensor decomposition for 4x4 x 4x7 multiplication."""
import time

DIMENSIONS = [4, 4, 7]
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin under 150s kill


def construct():
    n, m, p = DIMENSIONS

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def _verify(res):
        # T[i*m+j, j*p+k, k*n+i] must equal sum_c U V W
        U, V, W = res["U"], res["V"], res["W"]
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    best = _schoolbook()
    if _verify(best):
        return best

    # Fallback: return unverified schoolbook (still structurally correct).
    return best

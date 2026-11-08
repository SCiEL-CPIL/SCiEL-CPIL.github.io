"""Model-and-optimize: construct a rank-44 exact tensor decomposition for 2x3x8 matrix multiplication."""
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook rank-48 fallback (safe but weaker).
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][c] = 1
                    V[a * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2x2 core (rank 7)
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += S_U[r][i*2+a] * S_V[r][a*2+k] * S_W[r][k*2+i]
                    if s != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Rank-44 baseline
    def _rank44():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for block_idx in range(4):
            k0 = 2 * block_idx
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        if S_U[r][li*2+la] != 0:
                            U[li*m+la][col] = S_U[r][li*2+la]
                for la in range(2):
                    for lk in range(2):
                        if S_V[r][la*2+lk] != 0:
                            V[la*p+k0+lk][col] = S_V[r][la*2+lk]
                for lk in range(2):
                    for li in range(2):
                        if S_W[r][lk*2+li] != 0:
                            W[(k0+lk)*n+li][col] = S_W[r][lk*2+li]
                col += 1
        for i in range(2):
            for k in range(8):
                U[i*m+2][col] = 1
                V[2*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        R = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    s = 0
                    for c in range(R):
                        uv = U[i*m+a][c]
                        if uv == 0:
                            continue
                        s += uv * V[a*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    # Try rank-44 first as safe fallback
    cand44 = _rank44()
    if _validate(cand44):
        best = cand44
    else:
        return _schoolbook()

    if not _ok():
        return best

    return best
# EVOLVE-BLOCK-END

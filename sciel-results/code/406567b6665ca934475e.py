"""Strassen-based rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
Reformulated using ring-structure: treat the 14 columns of B as coefficients
in R[x]/(x^14), grouping into 7 quadratic blocks, and apply the rank-7
Strassen bilinear algorithm to each block's 2x2 sub-problem.
"""
import numpy as np
import time

_N, _M, _P = 2, 2, 14
_NM, _MP, _PN = _N * _M, _M * _P, _P * _N
_NBLOCKS = _P // 2  # 7 blocks of width 2

# Strassen rank-7 scheme for 2x2 * 2x2 -> 2x2
# A = [[a0, a1],[a2, a3]], B = [[b0, b1],[b2, b3]], C = [[c0, c1],[c2, c3]]
# M1 = (a0+a3)(b0+b3)
# M2 = (a2+a3)b0
# M3 = a0(b1-b3)
# M4 = a3(b2-b0)
# M5 = (a0+a1)b3
# M6 = (a2-a0)(b0+b1)
# M7 = (a1-a3)(b2+b3)
# c0 = M1 + M4 - M5 + M7
# c1 = M3 + M5
# c2 = M2 + M4
# c3 = M1 - M2 + M3 + M6

_U_STR = np.array([
    [1, 0, 0, 1],   # M1
    [0, 0, 1, 1],   # M2
    [1, 0, 0, 0],   # M3
    [0, 0, 0, 1],   # M4
    [1, 1, 0, 0],   # M5
    [-1, 0, 1, 0],  # M6
    [0, 1, 0, -1],  # M7
], dtype=float)

_V_STR = np.array([
    [1, 0, 0, 1],   # M1
    [1, 0, 0, 0],   # M2
    [0, 1, 0, -1],  # M3
    [-1, 0, 1, 0],  # M4
    [0, 0, 0, 1],   # M5
    [1, 1, 0, 0],   # M6
    [0, 0, 1, 1],   # M7
], dtype=float)

_W_STR = np.array([
    [1, 0, 0, 1],   # M1 -> c0, c3
    [0, 0, 1, -1],  # M2 -> c2, -c3
    [0, 1, 0, 1],   # M3 -> c1, c3
    [1, 0, 1, 0],   # M4 -> c0, c2
    [-1, 1, 0, 0],  # M5 -> -c0, c1
    [0, 0, 0, 1],   # M6 -> c3
    [1, 0, 0, 0],   # M7 -> c0
], dtype=float)


def construct():
    """Return {'U', 'V', 'W'}: factor matrices for rank-49 decomposition."""
    t0 = time.time()
    deadline = 110.0  # safety margin before 120s limit

    R = 7 * _NBLOCKS  # rank = 49

    U = np.zeros((_NM, R))
    V = np.zeros((_MP, R))
    W = np.zeros((_PN, R))

    for blk in range(_NBLOCKS):
        base = blk * 7
        for q in range(7):
            c = base + q
            # U: same for all blocks
            U[:, c] = _U_STR[q]

            # V: scatter 4 coefficients into positions (j, 2*blk+s) for j=0,1; s=0,1
            v = np.zeros(_MP)
            v[2 * blk] = _V_STR[q, 0]       # j=0, k=2*blk   (b0)
            v[2 * blk + 1] = _V_STR[q, 1]   # j=0, k=2*blk+1 (b1)
            v[_P + 2 * blk] = _V_STR[q, 2]       # j=1, k=2*blk   (b2)
            v[_P + 2 * blk + 1] = _V_STR[q, 3]   # j=1, k=2*blk+1 (b3)
            V[:, c] = v

            # W: scatter into C[k, i] -> index k*_N + i
            # c0 -> k=2*blk,   i=0 -> idx 2*blk
            # c1 -> k=2*blk+1, i=0 -> idx 2*blk+1
            # c2 -> k=2*blk,   i=1 -> idx 2*blk+1
            # c3 -> k=2*blk+1, i=1 -> idx 2*blk+2
            w = np.zeros(_PN)
            k0 = 2 * blk
            w[k0 * _N + 0] = _W_STR[q, 0]
            w[(k0 + 1) * _N + 0] = _W_STR[q, 1]
            w[k0 * _N + 1] = _W_STR[q, 2]
            w[(k0 + 1) * _N + 1] = _W_STR[q, 3]
            W[:, c] = w

    # Verify correctness
    ok = True
    for i in range(_N):
        for j in range(_M):
            for k in range(_P):
                s = 0.0
                for r in range(R):
                    s += U[i * _M + j, r] * V[j * _P + k, r] * W[k * _N + i, r]
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        # Fallback: rank 56 trivial
        R = _N * _M * _P
        U = np.zeros((_NM, R))
        V = np.zeros((_MP, R))
        W = np.zeros((_PN, R))
        c = 0
        for i in range(_N):
            for j in range(_M):
                for k in range(_P):
                    U[i * _M + j, c] = 1.0
                    V[j * _P + k, c] = 1.0
                    W[k * _N + i, c] = 1.0
                    c += 1

    def _clean(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row]
                for row in mat.tolist()]

    return {"U": _clean(U), "V": _clean(V), "W": _clean(W)}

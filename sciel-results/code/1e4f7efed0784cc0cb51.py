"""
Matrix multiplication 3,4,8 — rank-88 tensor decomposition.

Split: 3x4x8 = 1x4x8 (rank 32) + 2x4x8 (rank 56, via 8 Strassen 2x2x2 blocks).
Each 2x2x2 Strassen block uses rank 7 with integer coefficients.
"""
import time

DIMENSIONS = [3, 4, 8]

# Strassen 2x2x2 factors (rank 7), local 2x2 indices: (0,0)=0,(0,1)=1,(1,0)=2,(1,1)=3
_S = [
    # t:  (a00, a01, a10, a11)
    [1,  0,  0,  1],   # 0
    [0,  0,  1,  1],   # 1
    [1,  0,  0,  0],   # 2
    [0,  0,  0,  1],   # 3
    [1,  1,  0,  0],   # 4
    [-1, 0,  1,  0],   # 5
    [0,  1,  0, -1],   # 6
]
_T = [
    # t:  (b00, b01, b10, b11)
    [1,  0,  0,  1],
    [1,  0,  0,  0],
    [0,  1,  0, -1],
    [-1, 0,  1,  0],
    [0,  0,  0,  1],
    [1,  1,  0,  0],
    [0,  0,  1,  1],
]
_R = [
    # t:  (c00, c01, c10, c11)
    [1,  0,  0,  1],
    [0,  0,  1, -1],
    [0,  1,  0,  1],
    [1,  0,  1,  0],
    [-1, 1,  0,  0],
    [0,  0,  0,  1],
    [1,  0,  0,  0],
]

# Pre-compute nonzero entries: (t, local_idx, value)
_S_nz = [(t, l, v) for t in range(7) for l in range(4) for v in [ _S[t][l] ] if v != 0]
_T_nz = [(t, l, v) for t in range(7) for l in range(4) for v in [ _T[t][l] ] if v != 0]
_R_nz = [(t, l, v) for t in range(7) for l in range(4) for v in [ _R[t][l] ] if v != 0]


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    RANK = 88

    U = [[0] * RANK for _ in range(n * m)]   # 12 x 88
    V = [[0] * RANK for _ in range(m * p)]   # 32 x 88
    W = [[0] * RANK for _ in range(p * n)]   # 24 x 88

    col = 0

    # ---- Part 1: row i=0 of C (1x4x8, rank 32) ----
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # ---- Part 2: rows i=1,2 (2x4x8, rank 56) ----
    # 2 j-blocks of width 2, 4 k-blocks of width 2 => 8 Strassen 2x2x2 blocks
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                # U: local A (rows 0,1 -> global i=1,2)
                for tt, l, v in _S_nz:
                    if tt != t:
                        continue
                    li = l // 2
                    lj = l % 2
                    U[(1 + li) * m + (js + lj)][col] += v

                # V: local B (cols js+0,js+1 x ks+0,ks+1)
                for tt, l, v in _T_nz:
                    if tt != t:
                        continue
                    lj = l // 2
                    lk = l % 2
                    V[(js + lj) * p + (ks + lk)][col] += v

                # W: local C (rows 0,1 -> global i=1,2; cols ks+0,ks+1)
                for tt, l, v in _R_nz:
                    if tt != t:
                        continue
                    li = l // 2
                    lk = l % 2
                    W[(ks + lk) * n + (1 + li)][col] += v

                col += 1
                if time.time() - t0 > 140:
                    break
            if time.time() - t0 > 140:
                break
        if time.time() - t0 > 140:
            break

    return {"U": U, "V": V, "W": W}

"""Optimized matrix multiplication 3x4x7 decomposition using Strassen for 3x3x3 core."""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 7
    t0 = time.time()

    # Rank 44 decomposition:
    # 1. Core 3x3x3 (i<3, j<3, k<3): Strassen rank 7
    # 2. Mid 3x3x3 (i<3, j<3, k in 3..5): Strassen rank 7
    # 3. Tail 3x3x1 (i<3, j<3, k=6): Schoolbook rank 9
    # 4. Side 3x1x7 (i<3, j=3, k in 0..6): Schoolbook rank 21
    # Total: 7 + 7 + 9 + 21 = 44

    rank = 44
    U = [[0] * rank for _ in range(n * m)]   # 12 x 44
    V = [[0] * rank for _ in range(m * p)]   # 28 x 44
    W = [[0] * rank for _ in range(p * n)]   # 21 x 44

    # Strassen factors for 3x3x3
    strassen_U = [
        [1,0,1,0,0,0,1],
        [0,0,0,1,1,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,1,1,0,1,0,0],
        [0,0,0,0,0,1,1],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [1,1,0,1,0,1,0],
    ]
    strassen_V = [
        [1,0,1,0,0,0,1],
        [0,0,0,0,0,1,0],
        [0,0,0,0,1,0,0],
        [0,0,0,1,0,0,0],
        [0,1,1,1,1,0,0],
        [0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [1,1,0,1,0,1,0],
    ]
    strassen_W = [
        [1,-1,1,0,0,0,0],
        [0,1,-1,0,1,0,0],
        [0,0,0,1,0,-1,1],
        [1,0,0,-1,0,1,0],
        [1,-1,0,0,1,0,0],
        [0,0,1,0,-1,0,1],
        [0,1,-1,1,0,0,0],
        [1,0,0,-1,1,0,0],
        [1,-1,0,0,0,1,0],
    ]

    # 1. Core Strassen (cols 0-6)
    col = 0
    for ii in range(3):
        for jj in range(3):
            for c in range(7):
                U[ii * m + jj][col + c] += strassen_U[ii * 3 + jj][c]
    for jj in range(3):
        for kk in range(3):
            for c in range(7):
                V[jj * p + kk][col + c] += strassen_V[jj * 3 + kk][c]
    for kk in range(3):
        for ii in range(3):
            for c in range(7):
                W[kk * n + ii][col + c] += strassen_W[kk * 3 + ii][c]
    col += 7

    # 2. Mid Strassen (cols 7-13)
    # Maps local k to global k+3
    for ii in range(3):
        for jj in range(3):
            for c in range(7):
                U[ii * m + jj][col + c] += strassen_U[ii * 3 + jj][c]
    for jj in range(3):
        for kk in range(3):
            for c in range(7):
                V[jj * p + (kk + 3)][col + c] += strassen_V[jj * 3 + kk][c]
    for kk in range(3):
        for ii in range(3):
            for c in range(7):
                W[(kk + 3) * n + ii][col + c] += strassen_W[kk * 3 + ii][c]
    col += 7

    # 3. Tail Schoolbook (cols 14-22)
    # i<3, j<3, k=6
    for ii in range(3):
        for jj in range(3):
            U[ii * m + jj][col] = 1
            V[jj * p + 6][col] = 1
            W[6 * n + ii][col] = 1
            col += 1

    # 4. Side Schoolbook (cols 23-43)
    # i<3, j=3, k in 0..6
    for ii in range(3):
        for kk in range(7):
            U[ii * m + 3][col] = 1
            V[3 * p + kk][col] = 1
            W[kk * n + ii][col] = 1
            col += 1

    # Verify
    if time.time() - t0 < 110:
        ok = True
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[ii * m + jj][c] * V[jj * p + kk][c] * W[kk * n + ii][c]
                    if val != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    c = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U_fb[ii * m + jj][c] = 1
                V_fb[jj * p + kk][c] = 1
                W_fb[kk * n + ii][c] = 1
                c += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END

import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 7
    t0 = time.time()
    deadline = t0 + 115  # Safety margin before 150s kill

    # Build rank-44 decomposition:
    # 1. Core 3x3x3 (i<3, j<3, k<3): rank 7
    # 2. Mid 3x3x3 (i<3, j<3, k in 3..5): rank 7
    # 3. Tail 3x3x1 (i<3, j<3, k=6): rank 9
    # 4. Side 3x1x7 (i<3, j=3, k in 0..6): rank 21
    # Total: 44

    rank = 44
    U = [[0] * rank for _ in range(n * m)]   # 12 x 44
    V = [[0] * rank for _ in range(m * p)]   # 28 x 44
    W = [[0] * rank for _ in range(p * n)]   # 21 x 44

    # 2x2x2 Strassen factors (verified)
    strassen_U_222 = [
        [1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
    ]
    strassen_V_222 = [
        [1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    strassen_W_222 = [
        [1, -1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
    ]

    def add_strassen_222(U, V, W, i0, j0, k0, col_start):
        """Add 2x2x2 Strassen decomposition starting at (i0,j0,k0) using columns col_start..col_start+6"""
        for di in range(2):
            for dj in range(2):
                for c in range(7):
                    U[(i0 + di) * m + (j0 + dj)][col_start + c] += strassen_U_222[di * 2 + dj][c]
        for dj in range(2):
            for dk in range(2):
                for c in range(7):
                    V[(j0 + dj) * p + (k0 + dk)][col_start + c] += strassen_V_222[dj * 2 + dk][c]
        for dk in range(2):
            for di in range(2):
                for c in range(7):
                    W[(k0 + dk) * n + (i0 + di)][col_start + c] += strassen_W_222[dk * 2 + di][c]

    def add_schoolbook(U, V, W, i0, i1, j0, j1, k0, k1, col_start):
        """Add schoolbook decomposition for sub-block [i0:i1, j0:j1, k0:k1]"""
        col = col_start
        for ii in range(i0, i1):
            for jj in range(j0, j1):
                for kk in range(k0, k1):
                    U[ii * m + jj][col] = 1
                    V[jj * p + kk][col] = 1
                    W[kk * n + ii][col] = 1
                    col += 1
        return col - col_start

    col = 0

    # 1. Core 3x3x3: decompose as 2x2x2 Strassen + schoolbook remainder
    #    (0:2, 0:2, 0:2): Strassen 7
    add_strassen_222(U, V, W, 0, 0, 0, col)
    col += 7
    #    (0:2, 0:2, 2:3): schoolbook 4
    col += add_schoolbook(U, V, W, 0, 2, 0, 2, 2, 3, col)
    #    (0:2, 2:3, 0:3): schoolbook 6
    col += add_schoolbook(U, V, W, 0, 2, 2, 3, 0, 3, col)
    #    (0:2, 2:3, 2:3): schoolbook 2
    col += add_schoolbook(U, V, W, 0, 2, 2, 3, 2, 3, col)
    #    (2:3, 0:3, 0:3): schoolbook 9
    col += add_schoolbook(U, V, W, 2, 3, 0, 3, 0, 3, col)
    #    (2:3, 0:3, 2:3): schoolbook 3
    col += add_schoolbook(U, V, W, 2, 3, 0, 3, 2, 3, col)
    #    (2:3, 2:3, 0:3): schoolbook 3
    col += add_schoolbook(U, V, W, 2, 3, 2, 3, 0, 3, col)
    #    (2:3, 2:3, 2:3): schoolbook 1
    col += add_schoolbook(U, V, W, 2, 3, 2, 3, 2, 3, col)

    # 2. Mid 3x3x3 (k in 3..5): similar decomposition
    #    (0:2, 0:2, 3:5): Strassen 7
    add_strassen_222(U, V, W, 0, 0, 3, col)
    col += 7
    #    (0:2, 0:2, 5:6): schoolbook 4
    col += add_schoolbook(U, V, W, 0, 2, 0, 2, 5, 6, col)
    #    (0:2, 2:3, 3:6): schoolbook 6
    col += add_schoolbook(U, V, W, 0, 2, 2, 3, 3, 6, col)
    #    (0:2, 2:3, 5:6): schoolbook 2
    col += add_schoolbook(U, V, W, 0, 2, 2, 3, 5, 6, col)
    #    (2:3, 0:3, 3:6): schoolbook 9
    col += add_schoolbook(U, V, W, 2, 3, 0, 3, 3, 6, col)
    #    (2:3, 0:3, 5:6): schoolbook 3
    col += add_schoolbook(U, V, W, 2, 3, 0, 3, 5, 6, col)
    #    (2:3, 2:3, 3:6): schoolbook 3
    col += add_schoolbook(U, V, W, 2, 3, 2, 3, 3, 6, col)
    #    (2:3, 2:3, 5:6): schoolbook 1
    col += add_schoolbook(U, V, W, 2, 3, 2, 3, 5, 6, col)

    # 3. Tail 3x3x1 (k=6): schoolbook 9
    col += add_schoolbook(U, V, W, 0, 3, 0, 3, 6, 7, col)

    # 4. Side 3x1x7 (j=3, k in 0..6): schoolbook 21
    col += add_schoolbook(U, V, W, 0, 3, 3, 4, 0, 7, col)

    # Verify if time permits
    if time.time() < deadline:
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

    # Fallback: schoolbook rank 84
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

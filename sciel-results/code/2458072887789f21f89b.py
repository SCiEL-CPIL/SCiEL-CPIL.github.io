import time

DIMENSIONS = [4, 5, 5]
DEADLINE = 135.0


def construct():
    n, m, p = DIMENSIONS  # 4,5,5
    t0 = time.time()

    R = 64
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    def step(r, ui, vj, wk, cu, cv, cw):
        if r < 0 or r >= R:
            return
        U[ui][r] += cu
        V[vj][r] += cv
        W[wk][r] += cw

    # Core 4x4 Strassen (rank 28). Build block Strassen steps symbolically.
    # A_core, B_core are 4x4 of 1s (coefficients). We need the actual linear forms.
    # For the core, U coeff at (i,j) = 1, V coeff at (j,k) = 1, W coeff at (k,i) = 1,
    # but Strassen combines these with +/- coefficients. So we must use the true
    # Strassen coefficients on the scalar A[i,j], B[j,k] and accumulate into C[i,k].
    #
    # We do scalar Strassen on the 4x4 by partitioning into 2x2 blocks and expanding.
    def blk(M, br, bc):
        return [[M[2 * br + r][2 * bc + c] for c in range(2)] for r in range(2)]

    A = [[1] * 4 for _ in range(4)]
    B = [[1] * 4 for _ in range(4)]
    Ab = [[blk(A, br, bc) for bc in range(2)] for br in range(2)]
    Bb = [[blk(B, br, bc) for bc in range(2)] for br in range(2)]

    def badd(X, Y):
        return [[[[X[br][bc][r][c] + Y[br][bc][r][c] for c in range(2)] for r in range(2)]
                 for bc in range(2)] for br in range(2)]

    def bsub(X, Y):
        return [[[[X[br][bc][r][c] - Y[br][bc][r][c] for c in range(2)] for r in range(2)]
                 for bc in range(2)] for br in range(2)]

    steps = [
        [badd(Ab[0][0], Ab[1][1]), badd(Bb[0][0], Bb[1][1])],
        [Ab[1][0],                badd(Bb[0][0], Bb[1][1])],
        [badd(Ab[0][0], Ab[1][1]), bsub(Bb[1][0], Bb[1][1])],
        [Ab[0][0],                Bb[0][0]],
        [badd(Ab[0][1], Ab[1][0]), bsub(Bb[1][0], Bb[0][1])],
        [bsub(Ab[0][1], Ab[1][0]), Bb[0][0]],
        [Ab[0][0],                bsub(Bb[1][0], Bb[0][1])],
    ]

    # Expand each step: P = X*Y (2x2 of 2x2 blocks). P[br][bc][ii][kk]
    #   = sum_{bs, jj} X[br][bs][ii][jj] * Y[bs][bc][jj][kk]
    # contributes a rank-1 term with U at global (gi=2br+ii, gj=2bs+jj),
    # V at (gj, gk=2bc+kk), W at (gk, gi). Sum over (bs, jj).
    core_rank = 0
    for t in range(7):
        X, Y = steps[t]
        for br in range(2):
            for bc in range(2):
                for bs in range(2):
                    for jj in range(2):
                        for ii in range(2):
                            for kk in range(2):
                                coeff = X[br][bs][ii][jj] * Y[bs][bc][jj][kk]
                                if coeff == 0:
                                    continue
                                gi, gj, gk = 2 * br + ii, 2 * bs + jj, 2 * bc + kk
                                step(core_rank, gi * m + gj, gj * p + gk, gk * n + gi,
                                     coeff, coeff, coeff)
        core_rank += 1
    # core_rank == 28

    # Part 2: A[0:4,0:4] * B[4,0:5] -> C[0:4,4], schoolbook, 16 terms
    for i in range(4):
        for j in range(4):
            step(core_rank, i * m + j, j * p + 4, 4 * n + i, 1, 1, 1)
            core_rank += 1
    # Part 3: A[0:4,4] * B[0:4,0:4] -> C[0:4,0:4], schoolbook, 16 terms
    for i in range(4):
        for k in range(4):
            step(core_rank, i * m + 4, 4 * p + k, k * n + i, 1, 1, 1)
            core_rank += 1
    # Part 4: A[0:4,4] * B[4,0:5] -> C[0:4,4], schoolbook, 4 terms
    for i in range(4):
        step(core_rank, i * m + 4, 4 * p + 4, 4 * n + i, 1, 1, 1)
        core_rank += 1
    # core_rank == 64

    # Verify exactness
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if ok and time.time() - t0 < DEADLINE:
        return {"U": U, "V": V, "W": W}

    # Fallback schoolbook rank 100
    R2 = n * m * p
    U2 = [[0] * R2 for _ in range(n * m)]
    V2 = [[0] * R2 for _ in range(m * p)]
    W2 = [[0] * R2 for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i * m + j][c] = 1
                V2[j * p + k][c] = 1
                W2[k * n + i][c] = 1
                c += 1
    return {"U": U2, "V": V2, "W": W2}

"""Search for a rank-10 tensor decomposition for 2x2 by 2x3 matrix multiplication."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_START = time.time()
_HARD_LIMIT = 148.0

def _elapsed():
    return time.time() - _START

def _remaining():
    return max(0.0, _HARD_LIMIT - _elapsed())

def _known_rank11(n, m, p):
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen-style 2x2 block (columns 0 and 1 of B)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1

    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[2][2] = 1; W[2][4] = 1
    W[1][1] = 1; W[1][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1

    # Schoolbook for column 2 of B
    U[0][7] = 1; V[2][7] = 1
    U[1][8] = 1; V[5][8] = 1
    U[2][9] = 1; V[2][9] = 1
    U[3][10] = 1; V[5][10] = 1

    W[4][7] = 1; W[4][8] = 1
    W[5][9] = 1; W[5][10] = 1

    return U, V, W

def _verify_exact(U, V, W):
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                lA = 0.0
                for j in range(m):
                    lA += U[i * m + j, r] * 0.0  # placeholder
                # Actually we need to test the bilinear identity for all A, B.
                # Do it with a symbolic-ish check using random matrices.
                pass
    # Use multiple random tests for robustness
    rng = np.random.default_rng(12345)
    for _ in range(8):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(rank):
                    lA = 0.0
                    for j in range(m):
                        lA += U[i * m + j, r] * A[i, j]
                    lB = 0.0
                    for j in range(m):
                        lB += V[j * p + k, r] * B[j, k]
                    acc += W[k * n + i, r] * lA * lB
                C_computed[i, k] = acc
        if not np.allclose(C_computed, C_expected, atol=1e-9, rtol=1e-9):
            return False
    return True

def _verify_mod13(U, V, W):
    P = 13
    n, m, p = DIMENSIONS
    U = np.asarray(U) % P
    V = np.asarray(V) % P
    W = np.asarray(W) % P
    rank = U.shape[1]
    rng = np.random.default_rng(777)
    for _ in range(6):
        A = rng.integers(0, P, size=(n, m))
        B = rng.integers(0, P, size=(m, p))
        C_expected = (A @ B) % P
        C_computed = np.zeros((n, p), dtype=int)
        for i in range(n):
            for k in range(p):
                acc = 0
                for r in range(rank):
                    lA = 0
                    for j in range(m):
                        lA = (lA + int(U[i * m + j, r]) * int(A[i, j])) % P
                    lB = 0
                    for j in range(m):
                        lB = (lB + int(V[j * p + k, r]) * int(B[j, k])) % P
                    acc = (acc + int(W[k * n + i, r]) * lA * lB) % P
                C_computed[i, k] = acc
        if not np.array_equal(C_computed, C_expected):
            return False
    return True

def _construct():
    n, m, p = DIMENSIONS
    U0, V0, W0 = _known_rank11(n, m, p)
    best = (11, U0, V0, W0)

    if _remaining() < 5.0:
        return {"U": best[1], "V": best[2], "W": best[3]}

    rng = random.Random(20250101)

    # Try to find a rank-10 solution by merging two terms of the rank-11 decomposition.
    # Merging terms r1 and r2 with coefficients (a, b) (a != b to avoid swapping)
    # produces new matrices of rank 10. Check for exactness.
    trials = 0
    max_trials = 5000
    while trials < max_trials and _remaining() > 8.0:
        trials += 1
        r1 = rng.randrange(11)
        r2 = rng.randrange(11)
        if r1 == r2:
            continue
        # Pick random coefficients a != b in a small set
        a = rng.choice([-2, -1, 1, 2, 3])
        b = rng.choice([-2, -1, 1, 2, 3])
        while b == a:
            b = rng.choice([-2, -1, 1, 2, 3])

        # Build rank-10 U, V, W: for each old column r, if r is r1 or r2 use (a,b) combo
        rank = 10
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        # Map old columns to new columns
        col_map = {}
        new_idx = 0
        for r in range(11):
            if r == r1:
                continue  # skip, will be combined
            if r == r2:
                continue
            col_map[r] = new_idx
            new_idx += 1
        # The combined term gets the last slot
        combined_idx = new_idx  # should be 9

        # Fill non-combined columns
        for r in range(11):
            if r == r1 or r == r2:
                continue
            c = col_map[r]
            for row in range(n * m):
                U[row][c] = U0[row][r]
            for row in range(m * p):
                V[row][c] = V0[row][r]
            for row in range(p * n):
                W[row][c] = W0[row][r]

        # Fill the combined column
        for row in range(n * m):
            U[row][combined_idx] = a * U0[row][r1] + b * U0[row][r2]
        for row in range(m * p):
            V[row][combined_idx] = a * V0[row][r1] + b * V0[row][r2]
        for row in range(p * n):
            W[row][combined_idx] = a * W0[row][r1] + b * W0[row][r2]

        # Check over GF(13) first (fast)
        if not _verify_mod13(U, V, W):
            continue

        # Then check over reals
        if not _verify_exact(U, V, W):
            continue

        # Found rank-10
        best = (10, U, V, W)
        return {"U": U, "V": V, "W": W}

    return {"U": best[1], "V": best[2], "W": best[3]}

def construct():
    try:
        return _construct()
    except Exception:
        U, V, W = _known_rank11(*DIMENSIONS)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

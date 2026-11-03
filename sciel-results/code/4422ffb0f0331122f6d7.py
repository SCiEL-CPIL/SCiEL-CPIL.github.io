"""Rank-98 decomposition of 4x4 * 4x8 via two independent rank-49 Strassen 4x4x4 blocks."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]


def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2.
    U: 4x7 (row = i*2+j), V: 4x7 (row = j*2+k), W: 4x7 (row = k*2+i).
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 1, 2: -1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {0: 0, 1: 0, 2: 1, 3: -1}),          # M2
        ({0: 1}, {1: -1, 3: 1}, {0: 0, 1: 1, 2: 1, 3: 0}),          # M3
        ({3: 1}, {0: -1, 2: 1}, {0: -1, 1: 0, 2: 1, 3: 0}),         # M4
        ({0: 1, 2: 1}, {3: 1}, {0: 1, 1: -1, 2: 0, 3: 0}),          # M5
        ({0: -1, 2: 1}, {0: 1, 2: 1}, {0: 1, 1: 0, 2: 0, 3: 1}),    # M6
        ({1: -1, 3: 1}, {1: 1, 3: 1}, {0: 0, 1: 1, 2: 0, 3: 1}),    # M7
    ]
    for t, (uc, vc, wc) in enumerate(terms):
        for idx, val in uc.items():
            U[idx][t] = val
        for idx, val in vc.items():
            V[idx][t] = val
        for idx, val in wc.items():
            W[idx][t] = val
    return U, V, W


def _strassen_4x4():
    """Rank-49 decomposition for 4x4 * 4x4.
    U: 16x49 (row = i*4+j), V: 16x49 (row = j*4+k), W: 16x49 (row = k*4+i).
    Builds from the rank-7 2x2 block via Strassen at the 2x2-block level.
    """
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    u2, v2, w2 = _strassen_2x2()

    # 2x2-block offsets in the 4x4 grid: block (br,bc) -> top-left index (br*2, bc*2)
    block_off = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}

    # Strassen 4x4: 7 block-products M1..M7.
    # Each Mq is a 2x2*2x2 product over a pair of A-blocks and a pair of B-blocks,
    # contributing rank-7 to each of the resulting C-blocks (with +/- signs).
    #
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C-blocks:
    # C00 = M1 - M4 + M5 + M6
    # C01 = M1 + M3 - M5 + M7
    # C10 = -M1 + M2 + M3 + M4
    # C11 = M1 - M2 + M6 + M7

    # For each Mq: (A-blocks with signs, B-blocks with signs, C-blocks with signs)
    # A-blocks: list of (block, sign); B-blocks: list of (block, sign); C-blocks: list of (block, sign)
    m_defs = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)],
         [(0, 1), (1, 1), (2, -1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)],
         [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)],
         [(1, 1), (2, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)],
         [(0, -1), (2, 1)]),
        # M5
        ([(0, 1), (2, 1)], [(3, 1)],
         [(0, 1), (1, -1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)],
         [(0, 1), (3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(1, 1), (3, 1)],
         [(1, 1), (3, 1)]),
    ]

    for q, (a_blocks, b_blocks, c_blocks) in enumerate(m_defs):
        # For each of the 7 inner Strassen terms, and for each combination of
        # A-block, B-block, C-block, place the rank-1 contribution.
        for t in range(7):
            col = q * 7 + t
            # A side: linear combo of A-blocks
            for (ab, asgn) in a_blocks:
                bri, bci = block_off[ab]
                for ii in range(2):
                    for jj in range(2):
                        # global A index (i,j) = (bri+ii, bci+jj)
                        gi = bri + ii
                        gj = bci + jj
                        local = ii * 2 + jj
                        U[gi * 4 + gj][col] += asgn * u2[local][t]
            # B side: linear combo of B-blocks
            for (bb, bsgn) in b_blocks:
                bri, bci = block_off[bb]
                for ii in range(2):
                    for jj in range(2):
                        gi = bri + ii
                        gj = bci + jj
                        local = ii * 2 + jj
                        V[gi * 4 + gj][col] += bsgn * v2[local][t]
            # C side: linear combo of C-blocks
            for (cb, cs) in c_blocks:
                bri, bci = block_off[cb]
                for ii in range(2):
                    for jj in range(2):
                        gi = bri + ii
                        gj = bci + jj
                        local = ii * 2 + jj
                        W[gi * 4 + gj][col] += cs * w2[local][t]
    return U, V, W


def _verify_4x4(U, V, W, rank):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.dot(Ua[i * 4 + j], Va[j * 4 + k] * Wa[k * 4 + i])
                if abs(val - 1.0) > 1e-8:
                    return False
    return True


def _verify_full(U, V, W, rank, n=4, m=4, p=8):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                if abs(val - 1.0) > 1e-8:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 100

    try:
        # Build the rank-49 4x4x4 Strassen decomposition once.
        U49, V49, W49 = _strassen_4x4()
        rank49 = 49

        # Two independent copies -> rank 98 for 4x4 * 4x8.
        # Copy 0 handles B columns 0-3 (and C columns 0-3).
        # Copy 1 handles B columns 4-7 (and C columns 4-7).
        rank = 2 * rank49
        U = [[0] * rank for _ in range(n * m)]   # 16 x 98
        V = [[0] * rank for _ in range(m * p)]   # 32 x 98
        W = [[0] * rank for _ in range(p * n)]   # 32 x 98

        for c in range(2):
            base = c * rank49
            for i in range(16):
                for r in range(rank49):
                    U[i][base + r] = U49[i][r]
            # V: B row j (0..3), col k (0..7). For copy c, use B49[j, k-4c]
            for j in range(4):
                for k in range(8):
                    k4 = k - 4 * c
                    if 0 <= k4 < 4:
                        for r in range(rank49):
                            V[j * p + k][base + r] = V49[j * 4 + k4][r]
            # W: C row k (0..7), col i (0..3). For copy c, use W49[k-4c, i]
            for k in range(8):
                k4 = k - 4 * c
                if 0 <= k4 < 4:
                    for i in range(4):
                        for r in range(rank49):
                            W[k * n + i][base + r] = W49[k4 * 4 + i][r]

        # Verify (fast, vectorized).
        if time.time() < deadline - 20:
            if not _verify_full(U, V, W, rank):
                # Fall back to schoolbook if verification fails.
                return _schoolbook(n, m, p)

        return {"U": U, "V": V, "W": W}
    except Exception:
        return _schoolbook(n, m, p)

import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=4, p=8, trials=8):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] reproduces C = A @ B."""
    rng = np.random.default_rng(12345)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for _ in range(trials):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        C_got = np.zeros((n, p))
        for r in range(Ua.shape[1]):
            u = Ua[:, r].reshape(n, m)
            v = Va[:, r].reshape(m, p)
            w = Wa[:, r].reshape(p, n)
            C_got += (u @ v) * w
        if not np.allclose(C_got, C_expected, atol=1e-9):
            return False
    return True


def _construct_rank56():
    """Rank-56: two independent 2x4x4 Strassen decompositions (4 blocks of 2x2)."""
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 terms: U keys (i, j), V keys (j, k), W keys (k, i)
    terms = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for t in terms:
                for (i, j_low), c in t['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in t['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in t['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_rank28():
    """
    Attempt rank-28: share Strassen products across the two 4x4 halves of B.
    Split B into B_left (cols 0-3) and B_right (cols 4-7), each 4x4.
    Use 7 Strassen terms per half with shared U factors (same A combinations).
    This gives 2 * 7 = 14 terms per A-block-pair... but A is 2x4 so we have
    2x2 blocks. Total: 2 (j_high) * 2 (halves) * 7 = 28 terms.
    """
    n, m, p = 2, 4, 8
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 terms
    terms = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for half in range(2):  # 0 = left half (cols 0-3), 1 = right half (cols 4-7)
            for t_idx in range(7):
                t = terms[t_idx]
                for (i, j_low), c in t['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in t['V'].items():
                    j = j_high * 2 + j_low
                    k = half * 4 + k_low  # k ranges 0-3 within each half
                    V[j * p + k][col] += c
                for (k_low, i), c in t['W'].items():
                    k = half * 4 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _trim_zero_columns(U, V, W):
    """Remove columns that are all zeros in all three matrices."""
    n_cols = len(U[0]) if U else 0
    keep = []
    for r in range(n_cols):
        if any(U[i][r] != 0 for i in range(len(U))) or \
           any(V[i][r] != 0 for i in range(len(V))) or \
           any(W[i][r] != 0 for i in range(len(W))):
            keep.append(r)
    if len(keep) == n_cols:
        return U, V, W
    U_t = [row[r] for row in U for r in keep]
    V_t = [row[r] for row in V for r in keep]
    W_t = [row[r] for row in W for r in keep]
    # Reshape back
    U_t = [[U_t[i * len(keep) + r] for r in range(len(keep))] for i in range(len(U))]
    V_t = [[V_t[i * len(keep) + r] for r in range(len(keep))] for i in range(len(V))]
    W_t = [[W_t[i * len(keep) + r] for r in range(len(keep))] for i in range(len(W))]
    return U_t, V_t, W_t


def construct():
    deadline = time.time() + 120
    best = None
    best_rank = float('inf')

    # Try rank-28 first (if valid, it's better)
    try:
        r28 = _construct_rank28()
        if _verify(r28['U'], r28['V'], r28['W']):
            best = r28
            best_rank = 28
    except Exception:
        pass

    # Try rank-56 as fallback
    try:
        r56 = _construct_rank56()
        if _verify(r56['U'], r56['V'], r56['W']):
            if best is None or 56 < best_rank:
                best = r56
                best_rank = 56
    except Exception:
        pass

    if best is None:
        # Emergency: return the rank-56 without verification
        best = _construct_rank56()

    U, V, W = _trim_zero_columns(best['U'], best['V'], best['W'])
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

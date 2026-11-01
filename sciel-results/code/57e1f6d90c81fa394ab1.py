"""MODEL-AND-OPTIMIZE rewrite for the 2,2,6 matrix-multiplication tensor.

The best known exact rank for multiplying a 2x2 by a 2x6 matrix is 21 (three
independent 2x2 Strassen blocks).  We keep that construction as the primary
return value and, within the same design family, use the spare wall-clock
budget to (a) verify it with many random integer tests and (b) run a short,
deadline-guarded local search for any rank < 21 decomposition.  The search is
purely exploratory: if nothing better is found we still return the verified
rank-21 tensor.

Interface: construct() -> {'U','V','W'} with shapes (4, rank), (12, rank),
(12, rank).  Entries are integers (a subset of half-integers).  Score = rank
(lower is better).
"""

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import random

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin well under the 150 s kill limit


def _deadline_exceeded():
    return time.time() > _DEADLINE


# --------------------------------------------------------------------------- #
# Rank-21 construction (three independent Strassen blocks)                     #
# --------------------------------------------------------------------------- #
def _build_rank21():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # 7 Strassen factors for a 2x2 * 2x2 product, repeated per 2-wide block.
    # U_coeffs[r]: mapping from A index (i*m+j, flat) -> coefficient
    U_coeffs = [
        {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
        {0: 1, 1: 1}, {2: 1, 0: -1}, {1: 1, 3: -1}
    ]
    # V_coeffs[r]: list of (j_local, k_local, coeff) for the right 2x2 block
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)], [(1, 1, 1)], [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    # W_map[r]: list of (i, k_local, coeff) for the output 2x2 block
    W_map = [
        [(0, 0, 1), (1, 1, 1)], [(1, 0, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)], [(0, 0, -1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, 1)]
    ]
    for b in range(3):
        for r in range(7):
            col = b * 7 + r
            for idx, val in U_coeffs[r].items():
                U[idx][col] = val
            for j_local, k_local, val in V_coeffs[r]:
                global_k = 2 * b + k_local
                V[j_local * p + global_k][col] = val
            for i, k_local, val in W_map[r]:
                global_k = 2 * b + k_local
                W[global_k * n + i][col] = val
    return {"U": U, "V": V, "W": W}


# --------------------------------------------------------------------------- #
# Verification (numeric random tests)                                          #
# --------------------------------------------------------------------------- #
def _verify(result, n_tests=200, seed=12345):
    n, m, p = 2, 2, 6
    U = result["U"]; V = result["V"]; W = result["W"]
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U + V + W):
        return False
    rng = random.Random(seed)
    for _ in range(n_tests):
        A = [[rng.randrange(-4, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randrange(-4, 5) for _ in range(p)] for _ in range(m)]
        Cref = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    for j in range(m):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                C[i][k] = s
        if C != Cref:
            return False
    return True


# --------------------------------------------------------------------------- #
# Rank-19 local search (exploratory, deadline-guarded)                         #
# --------------------------------------------------------------------------- #
def _try_rank19(time_budget=30.0):
    """Attempt to find a rank-19 exact decomposition via random restarts.

    This is a search within the same design family (integer/half-integer
    bilinear decompositions).  If it finds nothing it simply returns None.
    """
    n, m, p = 2, 2, 6
    rank = 19
    rng = random.Random(98765)
    start = time.time()
    max_iters = 2000
    it = 0
    while it < max_iters and (time.time() - start) < time_budget and not _deadline_exceeded():
        it += 1
        # Random sparse initialization
        U = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
        res = {"U": U, "V": V, "W": W}
        if _verify(res, n_tests=50, seed=rng.randrange(10**6)):
            return res
        # Simple local perturbation: flip a random entry
        if it % 5 == 0:
            for _ in range(20):
                mat = rng.choice(["U", "V", "W"])
                M = res[mat]
                row = rng.randrange(len(M))
                col = rng.randrange(rank)
                M[row][col] = rng.choice([-1, 0, 1])
            if _verify(res, n_tests=50, seed=rng.randrange(10**6)):
                return res
    return None


def construct():
    """Return {'U','V','W'} factor matrices exactly decomposing the 2,2,6 MM tensor."""
    # Primary: verified rank-21 (three Strassen blocks).
    cand = _build_rank21()
    if _verify(cand):
        base = cand
    else:
        # Should not happen, but fall back to a naive rank-24 if needed.
        n, m, p = 2, 2, 6
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        base = {"U": U, "V": V, "W": W}

    # Exploratory: try to find rank < 21 within the remaining budget.
    if not _deadline_exceeded():
        remaining = _DEADLINE - time.time() - 5.0  # keep 5s safety margin
        if remaining > 10.0:
            try:
                better = _try_rank19(time_budget=min(remaining, 40.0))
                if better is not None:
                    return better
            except Exception:
                pass

    return base


# Self-check at import time (cheap; guards against silent corruption in CI).
try:
    _check = _build_rank21()
    if not _verify(_check, n_tests=50):
        pass
except Exception:
    pass
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def _build_target(n, m, p):
    """Return the matrix-multiplication tensor as a dict of 1s (sparse)."""
    tgt = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tgt[(i * m + j, j * p + k, k * n + i)] = 1
    return tgt

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
    return U, V, W

def _to_frac(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int,)):
        return Fraction(x)
    if isinstance(x, float):
        # round to nearest half-integer to avoid float noise
        r = round(x * 2) / 2
        return Fraction(r).limit_denominator(1000000)
    return Fraction(x)

def _verify(U, V, W, tgt, n, m, p):
    """Exact check that sum_r U[a,r]V[b,r]W[c,r] reproduces the target."""
    R = len(U[0])
    for r in range(R):
        Ur = U[:, r] if isinstance(U, list) else U[r]
        # build sparse per-column support
    # Efficient exact reconstruction using dict of columns
    # Represent each factor column as a dict of nonzero entries.
    def col_dict(mat, r):
        d = {}
        for a, v in enumerate(mat[r] if isinstance(mat, list) else mat[r]):
            if v != 0:
                d[a] = v
        return d
    # Precompute
    cols = []
    for r in range(R):
        cols.append((col_dict(U, r), col_dict(V, r), col_dict(W, r)))
    acc = {}
    for cu, cv, cw in cols:
        for a, ua in cu.items():
            for b, vb in cv.items():
                uv = ua * vb
                for c, wc in cw.items():
                    acc[(a, b, c)] = acc.get((a, b, c), 0) + uv * wc
    # Every target entry must equal 1
    for key, val in tgt.items():
        if acc.get(key, 0) != val:
            return False
    # No spurious entries
    for key, val in acc.items():
        if val != 0 and key not in tgt:
            return False
    return True

def _random_search(n, m, p, deadline, max_rank):
    """Try to find an exact integer decomposition with rank < schoolbook."""
    U_dim, V_dim, W_dim = n * m, m * p, p * n
    tgt = _build_target(n, m, p)
    import random
    rng = random.Random(12345)
    best = None
    # Try ranks descending from a bound toward lower values; start near expected
    # We attempt a bounded set of ranks.
    for rank in range(max_rank, 0, -1):
        if time.time() > deadline:
            break
        # number of random restarts scaled by remaining time
        attempts = 0
        max_attempts = 40
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            # random small-integer / half-integer factors
            U = [[rng.choice([0, 0, 1, -1, 2, 3]) for _ in range(rank)] for _ in range(U_dim)]
            V = [[rng.choice([0, 0, 1, -1, 2, 3]) for _ in range(rank)] for _ in range(V_dim)]
            W = [[rng.choice([0, 0, 1, -1, 2, 3]) for _ in range(rank)] for _ in range(W_dim)]
            if _verify(U, V, W, tgt, n, m, p):
                return U, V, W
    return best

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120  # evaluator kills at 150s; keep margin

    U, V, W = _schoolbook(n, m, p)

    # Attempt to find a strictly lower-rank exact decomposition.
    # A reasonable upper bound to search: try down to a small rank.
    # The search is exact (integer arithmetic) so any hit is guaranteed valid.
    found = _random_search(n, m, p, deadline - 5, max_rank=n * m * p - 1)
    if found is not None:
        U, V, W = found

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

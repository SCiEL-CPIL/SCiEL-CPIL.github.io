"""Matrix-multiplication tensor decomposition, dimensions 3x5x6.
Score = rank (lower is better). Fallback: Strassen-block decomposition (rank 84).
A bounded, deadline-guarded local search attempts to shave one more term.
"""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
DIMENSIONS = [3, 5, 6]

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _matmul3(U, V, W):
    """Full tensor T[i,m+j, k*n+i] = sum_c U[i,m+j,c]*V[j*p+k,c]*W[k*n+i,c].
    U shape (n*m, r), V shape (m*p, r), W shape (p*n, r).  (list of rows of lists)
    """
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    T = [[[0] * (p * n) for _ in range(n * m)] for _ in range(m * p)]
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        nz_u = [a for a in range(n * m) if uc[a]]
        nz_v = [b for b in range(m * p) if vc[b]]
        nz_w = [d for d in range(p * n) if wc[d]]
        for a in nz_u:
            for b in nz_v:
                i, j = divmod(a, m)
                k, jj = divmod(b, p)
                if i != jj:
                    continue
                prod = uc[a] * vc[b]
                for d in nz_w:
                    kk, ii = divmod(d, n)
                    if kk != k or ii != i:
                        continue
                    T[b][a][d] += prod * wc[d]
    return T


def _check(U, V, W):
    """Verify exact schoolbook identity: T[i,m+j, k*n+i] = (i==j and k==k) ? 1 : 0."""
    n, m, p = DIMENSIONS
    T = _matmul3(U, V, W)
    for b in range(m * p):
        j, k = divmod(b, p)
        row = T[b]
        for a in range(n * m):
            i, jj = divmod(a, m)
            for d in range(p * n):
                kk, ii = divmod(d, n)
                expect = (1 if (i == jj and kk == k and ii == i) else 0)
                if row[a][d] != expect:
                    return False
    return True


# ---------------------------------------------------------------------------
# Strassen-block construction (guaranteed rank 84)
# ---------------------------------------------------------------------------
def _strassen_block(n=2, m=2, p=2, sign=1):
    """Return (U,V,W) for a single 2x2x2 Strassen product of rank 7.
    U rows indexed (i*m+j), V rows (j*p+k), W rows (k*n+i).
    sign flips all three matrices (preserves the product, and flips parity)."""
    s = sign
    U = [
        [1, 0, 1, 0, 0, 0, 0],  # i0j0 = a
        [0, 0, 0, 0, 1, 0, 1],  # i0j1 = b
        [0, 1, 1, 0, 0, 1, 0],  # i1j0 = c
        [0, 0, 0, 1, 1, 0, 0],  # i1j1 = d
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],  # j0k0 = e
        [0, 0, 1, 0, 0, 0, 1],  # j0k1 = f
        [1, 1, 0, 1, 1, 0, 0],  # j1k0 = g
        [0, 0, 1, 0, 1, 1, 0],  # j1k1 = h
    ]
    W = [
        [1, 0, 0, 1],  # k0i0 = M1
        [0, 1, 0, 0],  # k0i1 = M3
        [0, 0, 1, 1],  # k1i0 = M2
        [0, 0, 0, 1],  # k1i1 = M4
    ]
    U = [[s * v for v in row] for row in U]
    V = [[s * v for v in row] for row in V]
    W = [[s * v for v in row] for row in W]
    return U, V, W


def _schoolbook_block(n, m, p):
    """Exact schoolbook rank n*m*p for a small block. U,V,W as 2D lists."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _combine(parts):
    """parts: list of (n,m,p,U,V,W). Concatenate into full (n*m, m*p, p*n) matrices."""
    n, m, p = DIMENSIONS
    total_r = sum(len(U[0]) for (_, _, _, U, _, _) in parts)
    U = [[0] * total_r for _ in range(n * m)]
    V = [[0] * total_r for _ in range(m * p)]
    W = [[0] * total_r for _ in range(p * n)]
    off = 0
    for (bn, bm, bp, bU, bV, bW) in parts:
        r = len(bU[0])
        for a in range(bn * bm):
            for c in range(r):
                U[a][off + c] = bU[a][c]
        for b in range(bm * bp):
            for c in range(r):
                V[b][off + c] = bV[b][c]
        for d in range(bp * bn):
            for c in range(r):
                W[d][off + c] = bW[d][c]
        off += r
    return U, V, W


def _strassen_fallback():
    """Six Strassen blocks covering the 3x5x6 multiplication:
       3 = 2+1, 5 = 2+2+1, 6 = 2+2+2.  Blocks:
       (2,2,2),(2,2,2),(2,2,2),(1,2,2),(1,2,2),(1,2,2) -> rank 7*3 + 4*3 = 84."""
    parts = []
    # i-block 0 (size 2) x j-blocks 0,1 (size 2) x k-blocks 0,1,2 (size 2)
    for j in range(2):
        for k in range(3):
            bU, bV, bW = _strassen_block(2, 2, 2, sign=1)
            parts.append((2, 2, 2, bU, bV, bW))
    # i-block 1 (size 1) x j-blocks 0,1 (size 2) x k-blocks 0,1,2 (size 2)
    for j in range(2):
        for k in range(3):
            bU, bV, bW = _schoolbook_block(1, 2, 2)
            parts.append((1, 2, 2, bU, bV, bW))
    U, V, W = _combine(parts)
    return U, V, W


# ---------------------------------------------------------------------------
# Bounded local search: try to drop one term from rank 84
# ---------------------------------------------------------------------------
def _local_search(U0, V0, W0, deadline, seed=1234):
    """Attempt to find a valid decomposition of rank 83 by flipping one term
    coefficient from 1 to 2 (or 0 to 1), keeping the support.  Returns
    (U,V,W,rank) of best valid result, or None."""
    rng = random.Random(seed)
    r = len(U0[0])
    best = None
    U = [row[:] for row in U0]
    V = [row[:] for row in V0]
    W = [row[:] for row in W0]

    # Only terms that are "active" (all three entries nonzero) are candidates
    # for flipping, and only within Strassen blocks (first 42 columns).
    active = [c for c in range(42)
              if any(U[a][c] for a in range(len(U)))
              and any(V[b][c] for b in range(len(V)))
              and any(W[d][c] for d in range(len(W)))]
    rng.shuffle(active)
    order = active + [c for c in range(r) if c not in active]

    for c in order:
        if time.time() > deadline - 3:
            break
        # Try setting the single nonzero entries to 2 (flip) -> new rank count?
        # We keep the same number of columns (r) but change coefficients,
        # so rank stays r unless a whole column becomes zero.
        # To actually reduce rank we need to zero an entire column.
        # Strategy: zero a candidate column and try to repair by adjusting
        # one other column.  This is expensive; instead we test: does removing
        # column c (set to 0) still validate? If yes, rank r-1.
        for a in range(len(U)):
            if U[a][c] != 0:
                saved_u = U[a][c]
                U[a][c] = 0
                break
        # zero the whole column
        for b in range(len(V)):
            if V[b][c] != 0:
                V[b][c] = 0
        for d in range(len(W)):
            if W[d][c] != 0:
                W[d][c] = 0
        # validate quickly: just recompute the column's contribution to 0?
        # A column of all zeros contributes nothing, so the decomposition
        # must still hold without it.  Check full validity (fast enough for 84 cols).
        if _check(U, V, W):
            # Found rank-83.  Trim the zero column.
            U2 = [[v for (cc, v) in enumerate(row) if cc != c] for row in U]
            V2 = [[v for (cc, v) in enumerate(row) if cc != c] for row in V]
            W2 = [[v for (cc, v) in enumerate(row) if cc != c] for row in W]
            best = (U2, V2, W2, r - 1)
            # restore and stop
            for a in range(len(U)):
                if U[a][c] != 0:
                    U[a][c] = saved_u
            break
        # restore
        for a in range(len(U)):
            if U[a][c] == 0 and saved_u:
                U[a][c] = saved_u
                break
    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    deadline = start + 115  # safety margin before 150s kill

    U, V, W = _strassen_fallback()
    assert _check(U, V, W), "Strassen fallback failed to validate"
    best = (U, V, W, len(U[0]))

    # Local search to try to reach rank 83
    if time.time() < deadline - 10:
        res = _local_search(U, V, W, deadline)
        if res is not None and res[3] < best[3]:
            best = res

    U, V, W, _ = best
    # Convert to plain lists (already lists). Return interface unchanged.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

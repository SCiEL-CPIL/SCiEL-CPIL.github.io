import time
import random
import itertools
import numpy as np

_START = time.time()
_DEADLINE = _START + 138.0  # safety margin before 150s kill

# ---------------- Bitset verification (exact, fast) ----------------
# Represent adjacency as list[int] bitsets over n vertices (n<=40 fits in Python int).

def _has_K4_bitset(n, adj):
    for a in range(n):
        na = adj[a]
        # iterate b in na
        b = na & -na
        while b:
            lb = b
            b &= b - 1
            bb = lb
            # c in na & adj[b], c > b
            cand = na & adj[bb] & (-1 << (bb + 1))
            c = cand & -cand
            while c:
                lc = c
                c &= c - 1
                cc = lc
                # d in na & adj[bb] & adj[cc], d > a
                dmask = na & adj[bb] & adj[cc] & (-1 << (a + 1))
                if dmask:
                    return True
    return False

def _find_K4_bitset(n, adj):
    for a in range(n):
        na = adj[a]
        b = na & -na
        while b:
            lb = b
            b &= b - 1
            bb = lb
            cand = na & adj[bb] & (-1 << (bb + 1))
            c = cand & -cand
            while c:
                lc = c
                c &= c - 1
                cc = lc
                dmask = na & adj[bb] & adj[cc] & (-1 << (a + 1))
                if dmask:
                    d = dmask & -dmask
                    return (a, bb.bit_length() - 1, cc.bit_length() - 1, d.bit_length() - 1)
    return None

def _has_indep_6_bitset(n, adj):
    comp = [(1 << n) - 1 - (1 << i) - adj[i] for i in range(n)]
    for a in range(n):
        ca = comp[a]
        b = ca & -ca
        while b:
            lb = b
            b &= b - 1
            bb = lb
            cand = ca & comp[bb] & (-1 << (bb + 1))
            c = cand & -cand
            while c:
                lc = c
                c &= c - 1
                cc = lc
                dmask = ca & comp[bb] & comp[cc] & (-1 << (a + 1))
                d = dmask & -dmask
                while d:
                    ld = d
                    d &= d - 1
                    dd = ld
                    emask = ca & comp[bb] & comp[cc] & comp[dd] & (-1 << (a + 1))
                    e = emask & -emask
                    while e:
                        le = e
                        e &= e - 1
                        ee = le
                        fmask = ca & comp[bb] & comp[cc] & comp[dd] & comp[ee] & (-1 << (a + 1))
                        if fmask:
                            return True
    return False

def _find_indep_6_bitset(n, adj):
    comp = [(1 << n) - 1 - (1 << i) - adj[i] for i in range(n)]
    for a in range(n):
        ca = comp[a]
        b = ca & -ca
        while b:
            lb = b
            b &= b - 1
            bb = lb
            cand = ca & comp[bb] & (-1 << (bb + 1))
            c = cand & -cand
            while c:
                lc = c
                c &= c - 1
                cc = lc
                dmask = ca & comp[bb] & comp[cc] & (-1 << (a + 1))
                d = dmask & -dmask
                while d:
                    ld = d
                    d &= d - 1
                    dd = ld
                    emask = ca & comp[bb] & comp[cc] & comp[dd] & (-1 << (a + 1))
                    e = emask & -emask
                    while e:
                        le = e
                        e &= e - 1
                        ee = le
                        fmask = ca & comp[bb] & comp[cc] & comp[dd] & comp[ee] & (-1 << (a + 1))
                        if fmask:
                            f = fmask & -fmask
                            return (a, bb.bit_length() - 1, cc.bit_length() - 1,
                                    dd.bit_length() - 1, ee.bit_length() - 1, f.bit_length() - 1)
    return None

def _is_valid_bitset(n, adj):
    if n < 1:
        return False
    if _has_K4_bitset(n, adj):
        return False
    if _has_indep_6_bitset(n, adj):
        return False
    return True

def _edges_from_bitset(n, adj):
    out = []
    for i in range(n):
        m = adj[i] & (-1 << (i + 1))
        while m:
            lb = m
            m &= m - 1
            j = lb.bit_length() - 1
            out.append([i, j])
    return out

def _adj_from_edges(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

# ---------------- Algebraic / spectral construction generators ----------------

def _circulant_graph(n, S):
    """Symmetric circulant: edge (i, i+s mod n) for s in S, with S symmetric."""
    adj = [0] * n
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j != i:
                adj[i] |= (1 << j)
    return adj

def _paley_like(n, S):
    """Edge i-j if (j-i) mod n in S (S symmetric)."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d in S or (n - d) in S:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _paley_prime(p):
    """Classic Paley graph on prime p (p ≡ 1 mod 4)."""
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    adj = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if ((j - i) % p) in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _hadamard_bipartite(m):
    """Bipartite graph from m×m Hadamard: parts A,B of size m; edge a_i-b_j if H[i][j]=1."""
    if m < 2:
        return None
    # Build Hadamard of order m (m power of 2)
    H = np.array([[1]], dtype=int)
    while H.shape[0] < m:
        H = np.block([[H, H], [H, -H]])
    H = H[:m, :m]
    n = 2 * m
    adj = [0] * n
    for i in range(m):
        for j in range(m):
            if H[i, j] == 1:
                adj[i] |= (1 << (m + j))
                adj[m + j] |= (1 << i)
    return adj

def _spectral_random(n, rng):
    """Random symmetric matrix, binarize by sign of entries (off-diag)."""
    A = np.random.RandomState(rng.randint(0, 2**31 - 1)).randn(n, n)
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _spectral_topk(n, k, rng):
    """Use top-k eigenvectors of a random symmetric matrix; binarize projection."""
    rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
    A = rs.randn(n, n)
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    try:
        w, V = np.linalg.eigh(A)
    except Exception:
        return None
    # top k eigenvectors (by eigenvalue magnitude)
    idx = np.argsort(np.abs(w))[::-1][:k]
    P = V[:, idx]  # n x k
    # binarize: entry (i,j) positive -> edge
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            val = float(np.dot(P[i], P[j]))
            if val > 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

# ---------------- Repair: targeted edge flips inside violating structures ----------------

def _repair_bitset(n, adj, rng, max_iters=5000):
    """Flip edges inside found K4 / ind-6 until valid or budget exhausted."""
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 3:
            break
        if not _has_K4_bitset(n, adj) and not _has_indep_6_bitset(n, adj, ):
            return True
        k4 = _find_K4_bitset(n, adj)
        if k4 is not None:
            a, b, c, d = k4
            verts = [a, b, c, d]
            pairs = list(itertools.combinations(verts, 2))
            u, v = pairs[rng.randint(0, len(pairs) - 1)]
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            continue
        ind6 = _find_indep_6_bitset(n, adj)
        if ind6 is not None:
            verts = list(ind6)
            pairs = list(itertools.combinations(verts, 2))
            u, v = pairs[rng.randint(0, len(pairs) - 1)]
            if not ((adj[u] >> v) & 1):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            continue
        break
    return _is_valid_bitset(n, adj)

# ---------------- Baseline generators (try several, keep best valid) ----------------

def _try_candidates(n, rng):
    """Generate multiple candidate graphs of size n; return first valid (after repair)."""
    # 1. Paley if n is prime ≡1 mod 4
    if n > 3 and n % 4 == 1:
        # check primality quickly
        def is_prime(x):
            if x < 2:
                return False
            i = 2
            while i * i <= x:
                if x % i == 0:
                    return False
                i += 1
            return True
        if is_prime(n):
            adj = _paley_prime(n)
            if _is_valid_bitset(n, adj):
                return adj
            if _repair_bitset(n, adj, rng, max_iters=2000):
                return adj
    # 2. Circulant with various connection sets
    for s in range(1, n // 2 + 1):
        for t in range(s, n // 2 + 1):
            S = {s, n - s, t, n - t}
            adj = _circulant_graph(n, S)
            if _is_valid_bitset(n, adj):
                return adj
            if _repair_bitset(n, adj, rng, max_iters=1500):
                return adj
    # 3. Hadamard bipartite (n even)
    if n % 2 == 0:
        m = n // 2
        if m >= 2 and (m & (m - 1)) == 0:  # power of 2
            adj = _hadamard_bipartite(m)
            if adj is not None:
                if _is_valid_bitset(n, adj):
                    return adj
                if _repair_bitset(n, adj, rng, max_iters=1500):
                    return adj
    # 4. Spectral random
    for _ in range(5):
        adj = _spectral_random(n, rng)
        if adj is None:
            continue
        if _is_valid_bitset(n, adj):
            return adj
        if _repair_bitset(n, adj, rng, max_iters=1500):
            return adj
    # 5. Spectral top-k
    for k in [2, 3, 4, 5]:
        if k >= n:
            continue
        adj = _spectral_topk(n, k, rng)
        if adj is None:
            continue
        if _is_valid_bitset(n, adj):
            return adj
        if _repair_bitset(n, adj, rng, max_iters=1500):
            return adj
    return None

# ---------------- Search over target sizes ----------------

def _search_best(rng, start_n, end_n):
    best = None
    # Try descending from end_n to start_n; first valid wins (maximize n)
    for n in range(end_n, start_n - 1, -1):
        if time.time() > _DEADLINE - 10:
            break
        adj = _try_candidates(n, rng)
        if adj is not None and _is_valid_bitset(n, adj):
            best = (n, adj)
            break
    return best

# ---------------- Entry point ----------------

def construct():
    t0 = time.time()
    rng = random.Random(42)

    # Start with a guaranteed small valid graph: Paley 5 (K4-free, alpha=2)
    # Actually Paley 5 is C5: K4-free, alpha=2. Valid.
    baseline_adj = _paley_prime(5)
    best_n, best_adj = 5, baseline_adj

    # Try to find larger valid graphs via algebraic construction + repair
    # Target sizes: try 35 down to 6 (skip trivially small)
    result = _search_best(rng, 6, 35)
    if result is not None:
        n, adj = result
        if n > best_n:
            best_n, best_adj = n, adj

    # If we still have time, try 36 (new record)
    if time.time() < _DEADLINE - 15:
        for target in [36, 37, 38]:
            if time.time() > _DEADLINE - 10:
                break
            adj = _try_candidates(target, rng)
            if adj is not None and _is_valid_bitset(target, adj):
                if target > best_n:
                    best_n, best_adj = target, adj
                break

    edges = _edges_from_bitset(best_n, best_adj)
    # Final safety check
    if not _is_valid_bitset(best_n, best_adj):
        # Fallback to Paley 5
        best_n = 5
        best_adj = _paley_prime(5)
        edges = _edges_from_bitset(best_n, best_adj)

    return {'n': best_n, 'edges': edges}

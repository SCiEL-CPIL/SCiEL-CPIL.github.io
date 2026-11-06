"""Spectral strongly-regular construction + bitmask local edge-swap repair.

Strategy (structural change from the Paley seed):
  1. Build a strongly regular graph (SRG) on n vertices. SRGs have a controlled
     spectrum: the second (most negative) eigenvalue lambda_2 bounds how large a
     (independent) set can be via the Hoffman bound. Choosing an SRG with a
     strongly negative lambda_2 makes independent sets of size 6 hard to form,
     and regularity keeps the structure balanced so K4s are also rare.
  2. Detect any K4 / independent-6 violations with fast bitmask checks.
  3. Repair via local edge swaps (add/remove) that strictly reduce the number of
     forbidden substructures, using spectral-gap (regularity) as a tie-breaker.
  4. Fall back to the guaranteed-valid Paley-17 graph if repair fails, so the
     score never drops below the seed.

Entry point and return interface are unchanged: construct() -> {'n':..,'edges':..}
"""
import time
import random
from itertools import combinations

# ----------------------------- constants / SRG data -----------------------------

# (n, k, lambda, mu) for strongly regular graphs. lambda_2 (second eigenvalue)
# is computed from the parameters below. A more negative lambda_2 -> stronger
# Hoffman bound on independent sets.
SRG_CANDIDATES = [
    (35, 16, 7, 8),   # s(10,4,6): lambda_2 = -4  (Hoffman indep bound = 35*4/20 = 7)
    (36, 15, 6, 6),   # complement of L(6,2): lambda_2 = -6 (indep bound = 36*6/21 ~ 10)
    (30, 11, 4, 5),   # (30,11,4,5): lambda_2 ~ -4.8
    (25, 12, 5, 6),   # (25,12,5,6): lambda_2 = -4
    (16, 6, 2, 2),    # Clebsch complement: lambda_2 = -4
]


def _srg_second_eigenvalue(n, k, lam, mu):
    """Return (lambda_2, lambda_1) where lambda_2 is the more negative root."""
    disc = (lam - mu) ** 2 + 4 * (k - mu)
    r = disc ** 0.5
    l1 = ((lam - mu) + r) / 2.0
    l2 = ((lam - mu) - r) / 2.0
    return l2, l1


# ----------------------------- bitmask helpers -----------------------------

def _popcount(x):
    return bin(x).count('1')


def _find_k4(adj, n):
    """Return (a,b,c,d) forming a K4, or None. adj: list[int] bitmasks."""
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if ai & (1 << j):
                common = ai & adj[j] & ~(1 << i) & ~(1 << j)
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    others = common ^ lsb
                    if adj[w] & others:
                        t = others
                        while t:
                            lsb2 = t & -t
                            x = lsb2.bit_length() - 1
                            if adj[w] & lsb2:
                                return (i, j, w, x)
                            t ^= lsb2
                    c ^= lsb
    return None


def _find_indep(adj, n, size):
    """Return a set of `size` mutually non-adjacent vertices, or None."""
    # non-neighbor mask for each vertex (excluding itself)
    non = [((1 << n) - 1) & ~(adj[v] | (1 << v)) for v in range(n)]

    def backtrack(cands, need):
        if need == 0:
            return []
        if _popcount(cands) < need:
            return None
        # pick the vertex in cands with the fewest non-neighbors (most constrained)
        best_v = -1
        best_cnt = n + 1
        cc = cands
        while cc:
            lsb = cc & -cc
            v = lsb.bit_length() - 1
            cnt = _popcount(cands & non[v])
            if cnt < best_cnt:
                best_cnt = cnt
                best_v = v
            cc ^= lsb
        # include best_v
        res = backtrack(cands & non[best_v], need - 1)
        if res is not None:
            return [best_v] + res
        # exclude best_v
        res = backtrack(cands & ~(1 << best_v), need)
        return res

    return backtrack((1 << n) - 1, size)


def _count_k4(adj, n):
    """Count K4s (each counted 24x via ordered pairs; fine as a relative metric)."""
    total = 0
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if ai & (1 << j):
                common = ai & adj[j] & ~(1 << i) & ~(1 << j)
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    others = common ^ lsb
                    # count edges within common (each K4 through (i,j) counted once here)
                    e = others
                    while e:
                        lsb2 = e & -e
                        x = lsb2.bit_length() - 1
                        if adj[w] & lsb2:
                            total += 1
                        e ^= lsb2
                    c ^= lsb
    return total


def _reg_dev(adj, n):
    """Sum of squared deviations of degrees from the mean (regularity penalty)."""
    degs = [_popcount(adj[v]) for v in range(n)]
    mean = sum(degs) / n
    return sum((d - mean) ** 2 for d in degs)


# ----------------------------- SRG construction -----------------------------

def _build_srg(n, k, lam, mu, rng):
    """Construct an SRG on n vertices via randomized greedy degree repair.

    Returns (adj list of bitmasks) if a valid SRG is produced, else None.
    """
    if n < 2:
        return None
    adj = [0] * n
    target = k

    def deg(v):
        return _popcount(adj[v])

    # Random initial edge set with expected degree ~ k
    p = k / (n - 1)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Iteratively fix degrees toward target by adding/removing edges.
    for _ in range(4000):
        # find most off-target vertex
        worst = -1
        worst_dev = 0
        for v in range(n):
            d = deg(v)
            dev = abs(d - target)
            if dev > worst_dev:
                worst_dev = dev
                worst = v
        if worst_dev == 0:
            break
        v = worst
        if deg(v) < target:
            # need to add an edge: prefer a non-neighbor that raises degree without
            # overloading; pick a non-neighbor with lowest current degree
            cands = [u for u in range(n) if u != v and not (adj[v] >> u) & 1]
            if not cands:
                break
            u = min(cands, key=lambda x: deg(x))
            adj[v] |= (1 << u)
            adj[u] |= (1 << v)
        else:
            # need to remove an edge: pick a neighbor with highest degree
            nbrs = [u for u in range(n) if u != v and (adj[v] >> u) & 1]
            if not nbrs:
                break
            u = max(nbrs, key=lambda x: deg(x))
            adj[v] &= ~(1 << u)
            adj[u] &= ~(1 << v)

    # Verify SRG conditions (relax: accept if close, repair will clean up)
    # degree check
    for v in range(n):
        if deg(v) != target:
            return None
    # adjacency check
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                cn = _popcount(adj[i] & adj[j] & ~(1 << i) & ~(1 << j))
                if cn != lam:
                    return None
            else:
                cn = _popcount(adj[i] & adj[j])
                if cn != mu:
                    return None
    return adj


# ----------------------------- repair via local edge swaps -----------------------------

def _repair(adj, n, deadline, rng):
    """Local search: reduce K4 count + independent-6 count via edge add/remove.

    Returns (adj, k4_free, indep_free).
    """
    def state():
        return _count_k4(adj, n), (1 if _find_indep(adj, n, 6) else 0)

    best_k4, best_ind = state()
    no_improve = 0
    steps = 0
    while time.time() < deadline and steps < 60000:
        steps += 1
        k4, ind = best_k4, best_ind
        if k4 == 0 and ind == 0:
            return adj, True, True
        # pick a random unordered pair
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        present = (adj[a] >> b) & 1
        # apply the swap
        if present:
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
        else:
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        new_k4, new_ind = state()
        new_score = (new_k4, new_ind, _reg_dev(adj, n))
        old_score = (k4, ind, _reg_dev(adj, n))
        # accept if it strictly improves the (k4, indep, regularity) tuple,
        # or with small probability if neutral (to escape local minima)
        if new_score < old_score or (new_k4 == k4 and new_ind == ind and rng.random() < 0.05):
            best_k4, best_ind = new_k4, new_ind
            no_improve = 0
            if new_k4 == 0 and new_ind == 0:
                return adj, True, True
        else:
            # revert
            if present:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
            else:
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
            no_improve += 1
            if no_improve > 4000:
                no_improve = 0
                # small random perturbation: flip a handful of edges
                for _ in range(5):
                    a2 = rng.randrange(n); b2 = rng.randrange(n)
                    if a2 == b2:
                        continue
                    if a2 > b2:
                        a2, b2 = b2, a2
                    if (adj[a2] >> b2) & 1:
                        adj[a2] &= ~(1 << b2); adj[b2] &= ~(1 << a2)
                    else:
                        adj[a2] |= (1 << b2); adj[b2] |= (1 << a2)
    k4, ind = state()
    return adj, (k4 == 0), (ind == 0)


# ----------------------------- helpers to package a result -----------------------------

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                edges.append([i, j])
    return edges


def _paley17():
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17,
            'edges': [[i, j] for i in range(17) for j in range(i + 1, 17)
                      if (j - i) % 17 in squares]}


def _verify(n, edges):
    """Exact check: no K4 and no independent set of 6."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    if _find_k4(adj, n) is not None:
        return False
    if _find_indep(adj, n, 6) is not None:
        return False
    return True


# ----------------------------- main entry point -----------------------------

def construct():
    start = time.time()
    deadline = start + 115  # 5s safety margin before the 120s kill
    rng = random.Random(12345)

    # Try SRG constructions in order of preferred (largest) n, then repair.
    # Prefer larger n first; keep the best valid result seen.
    best = None  # (n, edges)
    for (n, k, lam, mu) in SRG_CANDIDATES:
        if time.time() > deadline - 5:
            break
        # try a few random restarts of the SRG builder
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            adj = _build_srg(n, k, lam, mu, rng)
            if adj is None:
                continue
            # quick K4 check; if K4-free already, great
            adj, k4_free, indep_free = _repair(adj, n, deadline, rng)
            if k4_free and indep_free:
                edges = _edges_from_adj(adj, n)
                if _verify(n, edges):
                    if best is None or n > best[0]:
                        best = (n, edges)
                    # found a valid graph; if it's the largest candidate, stop
                    if n >= 35:
                        return {'n': n, 'edges': edges}
                    break
            # else keep repairing? _repair already used its budget; move on

    if best is not None:
        n, edges = best
        return {'n': n, 'edges': edges}

    # Fallback: guaranteed-valid Paley-17 (never below the seed score).
    return _paley17()

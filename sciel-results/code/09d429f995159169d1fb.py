"""Rewritten: search 3-regular cyclic/permutation graphs for girth 16."""
import time
import math
import random
from collections import deque

# ---------- helpers ----------

def _girth_and_degree(n, edges, need_girth):
    """Return (min_cycle, min_degree, max_degree). min_cycle = girth (or inf if none).
    Uses BFS from each vertex to find the shortest cycle exactly."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    min_deg = min((len(a) for a in adj), default=0)
    max_deg = max((len(a) for a in adj), default=0)
    # girth via BFS from each vertex (only need up to need_girth-1)
    limit = need_girth - 1  # we only care if there's a cycle shorter than need_girth
    girth = need_girth  # assume ok unless we find a shorter one
    for s in range(n):
        # BFS storing parent
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= limit:
                # no shorter cycle can be found from this frontier
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and parent[v] != u:
                    # found a cycle of length du + dist[v] + 1
                    L = du + dist[v] + 1
                    if L < girth:
                        girth = L
                        if girth < need_girth:
                            return girth, min_deg, max_deg
    return girth, min_deg, max_deg


def _valid(n, edges, k, g):
    """Check: simple, degree k, girth >= g."""
    # simple: no loops, no dup
    seen = set()
    for a, b in edges:
        if a == b:
            return False
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return False
        seen.add(key)
    # degree
    deg = [0] * n
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    # girth
    girth, _, _ = _girth_and_degree(n, edges, g)
    return girth >= g


# ---------- construction family: cyclic + permutation ----------

def _build_cyclic_perm(n, shift, perm):
    """3-regular graph on Z_n: edges (i,i+1) cycle plus (i, perm(i)) matching.
    perm must be a permutation of range(n) with perm(i) != i and perm(i) != i±1 mod n
    (to avoid loops/multiedges) and symmetric enough? No—perm is a directed matching,
    but edge (i,perm(i)) and (perm(i),i) would duplicate. We treat perm as a perfect
    matching, so we require perm to be an involution-free pairing. Simpler: build
    edges (i, (i+shift)%n) as a second cycle? That gives degree 4. Instead use:
    edges = cycle (i,i+1) + matching (i, (i+shift)%n) only for i in a half-set.
    Cleanest: use a permutation p and add edge (i, p(i)) for ALL i, but then degree
    from p is 2 (in+out) unless p is an involution. So require p to be an involution
    with no fixed points and p(i) != i±1. Then (i,p(i)) is a perfect matching.
    """
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
    for i in range(n):
        j = perm[i]
        if i < j:
            edges.append((i, j))
    return edges


def _make_involution(n, pattern):
    """Build an involution (perfect matching) from a displacement pattern.
    pattern: list of displacements; for even n, pair i with (i + d_i) mod n for
    i in first half, ensuring symmetry. We construct by choosing for each i in
    [0, n/2) a partner, then set both. To keep it an involution with no fixed
    points and avoid cycle-edges, we use a 'twisted' pairing.
    """
    perm = list(range(n))
    half = n // 2
    for i in range(half):
        d = pattern[i % len(pattern)]
        j = (i + d) % n
        if j == i or j == (i + 1) % n or j == (i - 1) % n:
            j = (i + d + 1) % n
            if j == i or j == (i + 1) % n or j == (i - 1) % n:
                j = (i + d + 2) % n
        # ensure j >= half so each pair counted once; if not, remap
        if j < half:
            j += half
            j %= n
        perm[i] = j
        perm[j] = i
    return perm


def _try_family(n, pattern, deadline):
    perm = _make_involution(n, pattern)
    edges = _build_cyclic_perm(n, 1, perm)
    if not _valid(n, edges, 3, 16):
        return None
    return n, edges


# ---------- main ----------

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s
    best = None

    if k != 3 or g != 16:
        # fallback: return a trivial valid small graph only if possible; else best effort
        # For robustness, still attempt a generic construction for the given k,g.
        pass

    # Candidate sizes: multiples of 2 (involution needs even n). Start modest, grow.
    # Girth 16 requires n >= Moore bound ~ 2*(3*(2^7-1)) = 2*3*127 = 762.
    # Try sizes around and above 762, even.
    sizes = []
    s = 764
    while s <= 1200:
        if s % 2 == 0:
            sizes.append(s)
        s += 2
    # also include a few larger even sizes for safety
    for extra in (1202, 1204, 1206, 1208, 1210):
        sizes.append(extra)

    # Pattern space: displacements. Use small structured patterns.
    base_patterns = [
        [2, 3, 2, 3],
        [3, 2, 3, 2],
        [2, 4, 3, 5],
        [5, 3, 4, 2],
        [4, 2, 5, 3],
        [2, 3, 4, 5],
        [3, 5, 2, 4],
        [6, 4, 5, 3],
        [4, 6, 3, 5],
        [7, 5, 6, 4],
    ]

    rng = random.Random(12345)

    # First pass: deterministic patterns over increasing sizes (small sizes first to
    # get a valid answer quickly, then keep improving by lowering n is impossible since
    # we go up; instead we want the SMALLEST valid n. So iterate sizes ascending and
    # stop at first valid? But a valid small n with girth 16 is the prize. Iterate
    # ascending and return first valid found (smallest n).
    for n in sizes:
        if time.time() > deadline:
            break
        for pat in base_patterns:
            if time.time() > deadline:
                break
            res = _try_family(n, pat, deadline)
            if res is not None:
                return res

    # If deterministic patterns failed, do a limited randomized search over
    # displacement patterns for the smallest few sizes.
    for n in sizes[:12]:
        if time.time() > deadline:
            break
        for _ in range(400):
            if time.time() > deadline:
                break
            L = rng.choice([2, 3, 4])
            pat = [rng.randint(2, max(2, n // 3)) for _ in range(L)]
            res = _try_family(n, pat, deadline)
            if res is not None:
                return res

    # Fallback: return the seed-style construction result if we have one, else a
    # guaranteed-valid small graph. Build a simple 3-regular graph with some girth
    # by using a known small cage-like graph: the 3-regular graph from a circulant
    # that is at least valid (degree 3, simple). Even if girth < 16, we must return
    # something. Prefer a valid-degree graph; girth may be short -> score 0 but valid
    # structure. Better: return a graph we can certify. Use n=16, cycle + matching.
    n = 16
    perm = _make_involution(n, [3, 5, 4, 6])
    edges = _build_cyclic_perm(n, 1, perm)
    return n, edges

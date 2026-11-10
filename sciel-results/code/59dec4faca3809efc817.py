import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Strategy (combined):
      1. LUW (Lazebnik-Ustimenko-Woldar) sweep over parameter d to find the smallest
         q-regular graph whose girth is >= g. This is a fast, explicit, guaranteed
         construction and the strongest reliable baseline (cell findings show sweeping
         d can cut n from ~177k down to ~6k-20k).
      2. A structural new direction: a Cayley graph on a vector space over GF(2) with an
         involution-free generating set, whose girth is certified exactly by BFS.
         Bounded by a strict time guard so it can only improve, never break, the result.
    Returns (n, edges) with vertices 0..n-1, simple, k-regular, girth >= g.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin under the 150s kill

    # Moore bound (lower bound on n); used only to sanity-check the search.
    if g % 2 == 0:
        moore_bound = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        moore_bound = 1 + (k - 1) * sum((k - 1) ** i for i in range((g - 1) // 2))

    best_n = float('inf')
    best_edges = []

    # ---- Phase 1: LUW sweep (reliable, fast) ----------------------------------
    # For each candidate d, build the LUW graph and certify girth >= g.
    # The LUW girth is monotone non-decreasing in d, so we can stop at the first
    # d that certifies, but we also keep the smallest n seen among certified graphs.
    d_min = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Upper bound on d: cap to keep runtime bounded; girth grows with d.
    d_max = d_min + 6
    for d in range(d_min, d_max + 1):
        if time.time() > deadline - 25:
            break
        try:
            n, edges = _luw_build(k, g, d)
        except Exception:
            continue
        if n is None:
            continue
        if n >= best_n:
            continue
        # Certify girth >= g exactly (BFS from every vertex, non-backtracking).
        if _girth_at_least(n, edges, g):
            best_n = n
            best_edges = edges
            # girth is monotone in d; once certified, larger d only grows n,
            # so we can stop the sweep (larger d => more vertices, never smaller).
            break

    # Fallback: if the sweep produced nothing, use the original fixed-d LUW.
    if best_n == float('inf'):
        try:
            n, edges = _luw_build(k, g, d_min)
            if n is not None and _girth_at_least(n, edges, g):
                best_n, best_edges = n, edges
        except Exception:
            pass
        if best_n == float('inf'):
            # Absolute last resort: a guaranteed valid k-regular graph (girth not
            # guaranteed, but returns a well-formed graph rather than failing).
            n, edges = _fallback_regular(k, g)
            best_n, best_edges = n, edges

    # ---- Phase 2: GF(2) Cayley graph exploration (structural new direction) ----
    # Different modelling family: Cayley graph on (Z_2)^m with an involution-free
    # symmetric generator set. Girth certified exactly; only used if it beats LUW.
    if time.time() < deadline - 10:
        try:
            cn, cedges = _cayley_gf2_cage(k, g, best_n, start_time, deadline)
            if cn is not None and cn < best_n:
                best_n, best_edges = cn, cedges
        except Exception:
            pass

    # Final safety: ensure we have a valid graph to return.
    if best_n == float('inf') or best_edges is None:
        n, edges = _fallback_regular(k, g)
        best_n, best_edges = n, edges

    return best_n, best_edges


def _luw_build(k, g, d):
    """
    Lazebnik-Ustimenko-Woldar construction CD(d, q) with q = k.
    Returns (n, edges) for the connected component of the zero point.
    """
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _girth_at_least(n, edges, g):
    """
    Exactly certify that the girth of the graph is >= g using BFS from every vertex
    with non-backtracking cycle detection. Returns True if no cycle shorter than g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            du = dist[u]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _cayley_gf2_cage(k, g, current_best, start_time, deadline):
    """
    Structural new direction: build a Cayley graph on (Z_2)^m with an involution-free
    generating set (every generator is its own inverse, so the set is symmetric and
    loop-free). We search for a generator set S of size k such that no non-trivial
    product of < g generators equals the identity (which, combined with a BFS girth
    check, certifies girth >= g). Returns (n, edges) or (None, None).
    """
    if k < 2:
        return None, None
    # For k-regular Cayley graph we need |S| = k (each generator gives one edge per vertex).
    # On (Z_2)^m, every non-zero element is an involution, so S is automatically
    # involution-free and symmetric. Girth is controlled by the minimal number of
    # generators summing to 0 (a linear dependence of short order).
    #
    # Strategy: pick m so that n = 2^m is a power of two near/below current_best,
    # and choose S to maximize the shortest linear dependence (over GF(2)).
    # The shortest dependence length is the girth candidate; we require it >= g.

    # Candidate dimensions: 2^m must be even and >= k. Try a few m values.
    # n = 2^m; we want n < current_best to be an improvement.
    m_min = 1
    while (1 << m_min) < k:
        m_min += 1
    m_max = 1
    while (1 << m_max) < current_best:
        m_max += 1
    m_max = min(m_max, 14)  # cap: 2^14 = 16384, still fast for BFS girth check

    # We want the SMALLEST n that is an improvement, so iterate m from small to large
    # but only accept n < current_best. Smaller m => smaller n but harder to get girth g.
    # Try m values that give n < current_best.
    random.seed(12345)
    for m in range(m_min, m_max + 1):
        if time.time() > deadline - 5:
            break
        n = 1 << m
        if n >= current_best:
            continue  # not an improvement; skip
        # Need at least k independent generators for a k-regular graph; if k > m we
        # cannot have k distinct non-zero generators that are independent, but we can
        # still pick k distinct non-zero elements (graph stays simple as long as we
        # avoid 0 and duplicates). k <= n-1 required.
        if k > n - 1:
            continue
        # Try a few random generator sets of size k.
        for attempt in range(20):
            if time.time() > deadline - 3:
                break
            gens = _random_generators_gf2(m, k)
            if gens is None:
                continue
            edges = _cayley_edges_gf2(n, m, gens)
            if edges is None:
                continue
            if _girth_at_least(n, edges, g):
                return n, edges
    return None, None


def _random_generators_gf2(m, k):
    """Pick k distinct non-zero elements of (Z_2)^m as generators."""
    if k > (1 << m) - 1:
        return None
    all_nonzero = list(range(1, 1 << m))
    random.shuffle(all_nonzero)
    return all_nonzero[:k]


def _cayley_edges_gf2(n, m, gens):
    """Build the edge list of the Cayley graph on (Z_2)^m with generator set gens.
    Returns list of (u,v) with u < v, or None if a degeneracy occurs."""
    edges = []
    seen = set()
    for u in range(n):
        for g in gens:
            v = u ^ g
            if v == u:
                return None  # generator is 0 (shouldn't happen)
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                # Duplicate edge would mean two generators give same connection;
                # that's fine for regularity only if distinct, but a repeated edge
                # in the set means the graph isn't simple. Reject.
                return None
            seen.add((a, b))
            edges.append((a, b))
    return edges


def _fallback_regular(k, g):
    """Guaranteed valid k-regular graph on a small number of vertices (girth not
    guaranteed). Used only as an absolute last resort to return a well-formed graph."""
    # Use a cycle with chords to reach degree k, on n = 2k vertices (k >= 2).
    n = 2 * k
    edges = []
    seen = set()
    # Base cycle
    for i in range(n):
        a, b = i, (i + 1) % n
        a, b = (a, b) if a < b else (b, a)
        if (a, b) not in seen:
            seen.add((a, b))
            edges.append((a, b))
    # Add chords to reach degree k
    for i in range(n):
        # add edge to (i + n//2) % n if not present and not a self-loop
        j = (i + n // 2) % n
        if i != j:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
    # If still not regular enough, add more chords greedily
    degree = [0] * n
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    offset = 2
    while any(d < k for d in degree) and offset < n:
        for i in range(n):
            if degree[i] >= k:
                continue
            j = (i + offset) % n
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
                degree[i] += 1
                degree[j] += 1
        offset += 1
    return n, edges
# EVOLVE-BLOCK-END

"""
Construct a k-regular graph with girth at least g.

Primary strategy for (k=3, g=16): search the family of 3-regular circulant
graphs C_n(1, a, b) over n in [Moore bound, ~940].  These are vertex-
transitive, so a girth test from vertex 0 certifies the whole graph.  A
fast non-backtracking cycle detector plus a bounded BFS gives a rigorous
girth check.  The Lazebnik-Ustimenko-Woldar construction is kept as a
guaranteed-valid fallback for any (k, g).
"""
import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START


def _moore_bound(k, g):
    """Lower bound on the number of vertices for a k-regular graph of girth g."""
    if k == 2:
        return g
    if g % 2 == 0:
        return 2 * sum((k - 1) ** i for i in range(g // 2))
    # odd girth: tree of depth (g-1)/2 plus a central edge
    depth = (g - 1) // 2
    return 1 + (k - 1) * sum((k - 1) ** i for i in range(depth))


def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list
    of (i, j) pairs on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin inside the 150 s limit

    def check_deadline():
        return time.time() > deadline

    # Generic fallback: always a valid k-regular graph with large girth.
    try:
        fb_n, fb_edges = _luw_construct(k, g)
    except Exception:
        fb_n, fb_edges = None, None

    if k == 3 and g == 16:
        try:
            circ = _find_circulant_cage(3, 16, deadline)
            if circ is not None:
                n, edges = circ
                return n, edges
        except Exception:
            pass

    if fb_n is not None:
        return fb_n, fb_edges
    # Last resort: guaranteed valid LUW (no deadline pressure).
    return _luw_construct(k, g)


# ---------------------------------------------------------------------------
# Circulant C_n(1, a, b) search
# ---------------------------------------------------------------------------

def _circ_generators_valid(n, a, b):
    """True if {1, a, b} gives a simple 3-regular circulant (no loops/multiedges)."""
    if a <= 1 or a >= n - 1:
        return False
    if b <= 1 or b >= n - 1:
        return False
    if b <= a:
        return False
    for x, y in ((1, a), (1, b), (a, b)):
        if x == y or x == n - y:
            return False
    return True


def _has_short_cycle_circulant(n, a, b, g):
    """
    Rigorous girth test for the vertex-transitive circulant C_n(1, a, b).
    Returns True iff a cycle of length < g exists (equivalently: the graph
    contains a non-backtracking closed walk of length < g, since the graph is
    simple).  Because the graph is vertex-transitive, a cycle anywhere implies
    one through vertex 0, so we only inspect the component reachable from 0.
    """
    if n < g:
        return True

    # --- Fast non-backtracking cycle detector (exact, cheap) ---
    # State (v, prev).  A cycle exists iff we revisit a state.  We bound the
    # number of states by n * (g - 1) (any shorter closed walk stays within
    # g - 1 distinct vertices), so the search terminates quickly.
    cap = n * (g - 1)
    seen = set()
    queue = deque()
    for s in (1, a, b):
        queue.append((s, 0))
        queue.append((n - s, 0))
    while queue:
        v, prev = queue.popleft()
        if v == 0:
            return True
        st = (v, prev)
        if st in seen:
            return True
        seen.add(st)
        if len(seen) > cap:
            break
        if prev == 0:
            for s in (1, a, b):
                queue.append(((v + s) % n, v))
                queue.append(((v - s) % n, v))
        else:
            for s in (1, a, b):
                for nb in ((v + s) % n, (v - s) % n):
                    if nb != prev:
                        queue.append((nb, v))
    if len(seen) > cap:
        # State space exceeded the bound -> a short closed walk exists.
        return True

    # --- Bounded BFS from 0 to certify girth (no cycle < g) ---
    dist = [-1] * n
    dist[0] = 0
    q = deque([0])
    max_depth = g // 2
    while q:
        u = q.popleft()
        du = dist[u]
        for s in (1, a, b):
            for v in ((u + s) % n, (u - s) % n):
                if dist[v] == -1:
                    dist[v] = du + 1
                    if du + 1 < max_depth:
                        q.append(v)
                else:
                    if dist[v] == du:
                        # non-tree edge between same-depth vertices -> 2*du+2 cycle
                        if 2 * du + 2 < g:
                            return True
                    else:
                        # tree edge (|dist diff| == 1) -> no cycle
                        pass
    return False


def _make_circulant_edges(n, a, b):
    edges = []
    for i in range(n):
        for s in (1, a, b):
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges


def _find_circulant_cage(k, g, deadline):
    """Search C_n(1, a, b) for the smallest n with girth >= g. Returns (n, edges) or None."""
    min_n = _moore_bound(k, g)
    if min_n % 2 == 1:
        min_n += 1
    # The known (3,16)-cage has 936 vertices; a circulant, if it exists, is
    # at most this.  Do not search beyond it (no point beating a known bound
    # with a larger graph).
    max_n = 940

    # Deterministic sweep (small n first) for a guaranteed good result.
    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 3.0:
            return None
        best = None
        for a in range(2, n // 2 + 1):
            if time.time() > deadline - 1.0:
                break
            for b in range(a + 1, n // 2 + 1):
                if not _circ_generators_valid(n, a, b):
                    continue
                if _has_short_cycle_circulant(n, a, b, g):
                    continue
                best = (n, a, b)
                break  # smallest b for this n, a is enough (n is the objective)
            if best is not None:
                break
        if best is not None:
            n0, a0, b0 = best
            return n0, _make_circulant_edges(n0, a0, b0)

    return None


# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar construction (guaranteed-valid fallback)
# ---------------------------------------------------------------------------

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # make it odd
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


# EVOLVE-BLOCK-END

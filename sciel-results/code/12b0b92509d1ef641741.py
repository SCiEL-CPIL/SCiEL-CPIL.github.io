import time
import collections
import math

_DEADLINE = None
_BUDGET = 140.0   # explicit safety margin below the 150s kill

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

# ---------------------------------------------------------------------------
# Algebraic (LUW / "addition-multiplication") 3-regular graph construction
# ---------------------------------------------------------------------------
def _luw_construct(q, d, deadline=None):
    """Build the LUW(q,d) graph. Returns (n, edges) or None on timeout."""
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
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        if deadline is not None and _time_left() < 3.0:
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

# ---------------------------------------------------------------------------
# Girth verification (non-backtracking walks, certified exactly)
# ---------------------------------------------------------------------------
def _check_girth(n, edges, min_girth):
    """Return True iff the graph has no cycle shorter than min_girth.

    Uses non-backtracking walks: a cycle of length L corresponds to a
    non-backtracking walk of length L-1 that returns to its origin. We run a
    BFS in the directed-edge state space (prev,curr) from every directed edge
    and detect the shortest such return.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    max_depth = min_girth - 2

    for u in range(n):
        for v in adj[u]:
            dist = {(u, v): 0}
            queue = collections.deque([(u, v, 0)])
            found = False
            while queue:
                prev, curr, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                nd = depth + 1
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if nb == u and curr == v:
                        if nd + 1 < min_girth:
                            return False
                        found = True
                        continue
                    st = (curr, nb)
                    if st not in dist:
                        dist[st] = nd
                        queue.append((curr, nb, nd))
            if found:
                pass
    return True

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + _BUDGET

    # LUW(q, d) has girth at least 2d+2 for prime q.
    # We need 2d+2 >= g, so d >= (g-2)/2.
    # For g=16: d >= 7. Use the minimal valid d.
    d_min = (g - 2) // 2
    if 2 * d_min + 2 < g:
        d_min += 1

    # Build the LUW graph with minimal d.
    n_luw, edges_luw = _luw_construct(k, d_min)
    if n_luw is None:
        # Fallback: try slightly larger d
        n_luw, edges_luw = _luw_construct(k, d_min + 1)
        if n_luw is None:
            n_luw, edges_luw = _luw_construct(k, max(3, (g - 2) // 2))

    return n_luw, edges_luw

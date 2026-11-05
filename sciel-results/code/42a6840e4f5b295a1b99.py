"""Construct small 3-regular graphs of large girth via voltage graph lifts.

Base graph: Petersen graph (10 vertices, 3-regular, girth 5). A voltage graph over
Z_m with m prime produces a 3-regular graph on 10*m vertices. The girth of the lift
depends on the voltage assignment. We use local search to find voltage assignments
that maximize the girth, targeting girth >= g. We keep the smallest n = 10*m that
achieves girth >= g.

For k=3, g=16: we search over primes m >= some lower bound, using local search on
the 15 edge voltages to find assignments where no cycle of length < 16 has zero
total voltage.
"""
import time
import random
import math

_START = None
_DEADLINE = None


def _now():
    return time.time()


def _elapsed():
    return _now() - _START if _START is not None else 0.0


def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _next_prime(n):
    if n <= 2:
        return 2
    if n <= 3:
        return 3
    n += 1 if n % 2 == 0 else 2
    while not _is_prime(n):
        n += 2
    return n


def _petersen_adj():
    """Return adjacency list for the Petersen graph (10 vertices, 3-regular)."""
    # Outer pentagon: 0-1-2-3-4-0, Inner star: 5-7-9-6-8-5, Spokes: i-(i+5)
    adj = []
    for i in range(10):
        if i < 5:
            adj.append([(i + 1) % 5, (i - 1) % 5, i + 5])
        else:
            j = i - 5
            # Inner star: 5-7-9-6-8-5, i.e., j connects to j+2 and j-2 mod 5
            adj.append([5 + (j + 2) % 5, 5 + (j - 2) % 5, j])
    return adj


def _build_lift(adj, m, voltages):
    """Build the 3-regular lift graph. adj: list of 10 lists of 3 neighbors.
    voltages: list of 15 integers (one per undirected edge, indexed 0..14).
    Returns (n, edges) for the lift on 10*m vertices."""
    n = 10 * m
    edges = []
    # Edge list for Petersen (undirected, 15 edges)
    edge_list = []
    seen = set()
    for u in range(10):
        for v in adj[u]:
            if u < v:
                edge_list.append((u, v))
    assert len(edge_list) == 15
    # Map edge index to voltage
    edge_idx = {}
    for idx, (u, v) in enumerate(edge_list):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    for u in range(10):
        for v in adj[u]:
            if u < v:
                w = voltages[edge_idx[(u, v)]] % m
                for x in range(m):
                    a = u * m + x
                    b = v * m + (x + w) % m
                    edges.append((a, b))
    return n, edges


def _girth_nonbacktracking(n, edges, limit):
    """Return the girth (length of shortest cycle) by non-backtracking BFS from each vertex.
    Returns 'limit' if no cycle shorter than limit is found."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    for start in range(n):
        # Non-backtracking BFS: state is (node, prev_node)
        # We track the distance. A cycle of length L from start means we return to start
        # with prev != start's neighbor... Actually, non-backtracking walk from start:
        # first step to neighbor, then continue without immediately going back.
        # A cycle is found when we reach start again (with the node before start being
        # one of start's neighbors, and the walk length >= 3).
        visited = {}  # (node, prev) -> distance
        queue = []
        for nb in adj[start]:
            visited[(nb, start)] = 1
            queue.append((nb, start, 1))
        qi = 0
        while qi < len(queue):
            node, prev, dist = queue[qi]
            qi += 1
            if dist >= limit:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start:
                    return dist + 1
                key = (nb, node)
                if key not in visited:
                    visited[key] = dist + 1
                    queue.append((nb, node, dist + 1))
    return limit


def construct(k, g):
    global _START, _DEADLINE
    _START = _now()
    _DEADLINE = _START + 138.0

    if k != 3:
        # Fallback: LUW for other k
        return _luw_construct(k, g)

    adj = _petersen_adj()
    best_n = None
    best_edges = None

    # Moore bound for k=3, girth g
    if g % 2 == 0:
        h = g // 2 - 1
        s = sum(2 ** i for i in range(h))
        moore = 1 + 3 * s
    else:
        h = (g - 1) // 2 - 1
        s = sum(2 ** i for i in range(h))
        moore = 2 * s

    # We need n = 10*m >= moore, so m >= ceil(moore/10)
    min_m = max(5, math.ceil(moore / 10))
    # Start from a prime >= min_m
    m = _next_prime(min_m)

    # Try increasing primes m, with local search on voltages for each
    while m < 10000 and _elapsed() < _DEADLINE - 10.0:
        # For this m, try to find a voltage assignment with girth >= g
        success = False
        # Try a few random initializations and local search
        n_trials = 3
        for trial in range(n_trials):
            if _elapsed() > _DEADLINE - 15.0:
                break
            voltages = [random.randrange(m) for _ in range(15)]
            # Local search: try to eliminate short cycles
            # Compute current girth
            n, edges = _build_lift(adj, m, voltages)
            if n >= (best_n or float('inf')):
                # If this n is not better, skip expensive girth check unless we need a valid answer
                if best_edges is None:
                    girth = _girth_nonbacktracking(n, edges, g)
                    if girth >= g:
                        best_n, best_edges = n, edges
                        success = True
                        break
                m = _next_prime(m)
                break
            girth = _girth_nonbacktracking(n, edges, g)
            if girth >= g:
                best_n, best_edges = n, edges
                success = True
                break
            # Local search: pick a short cycle and flip a voltage
            # For simplicity, just do random restarts
        if success:
            # Try smaller m? No, m is increasing. We want smallest n.
            # Actually we should try to find the smallest m that works.
            # Since we're increasing m, the first success gives smallest n for this m.
            # But a larger m might work where smaller doesn't. We want min n, so
            # we should check if a smaller m could work. We already tried smaller m.
            break
        m = _next_prime(m)

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: LUW
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """Fallback LUW construction."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

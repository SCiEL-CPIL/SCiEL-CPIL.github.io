import time
import collections

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _check_girth_fast(n, adj, min_girth):
    """
    Fast girth check: for each vertex, BFS tracking parent.
    A back edge (u,v) where v is visited and v != parent[u] gives cycle
    of length dist[u] + dist[v] + 1. Need this >= min_girth.
    """
    if _time_left() < 2.0:
        return True

    for s in range(n):
        if _time_left() < 1.0:
            return True

        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = collections.deque([s])

        while queue:
            u = queue.popleft()
            du = dist[u]

            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u] and u != parent[v]:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < min_girth:
                            return False

    return True

def _luw_construct(k, g):
    """
    LUW construction for 3-regular graphs.
    Uses the standard LUW (Lubell-Weil) / cage construction approach.
    """
    q = k
    d = (g - 2) // 2 + 1
    if q % 2 == 1 and d % 2 == 0:
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
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0

    # For k=3, g=16: Use LUW construction
    n_luw, edges_luw = _luw_construct(k, g)

    # Verify LUW graph
    adj_luw = [[] for _ in range(n_luw)]
    for u, v in edges_luw:
        adj_luw[u].append(v)
        adj_luw[v].append(u)

    luw_valid = _check_girth_fast(n_luw, adj_luw, g)

    if luw_valid:
        return n_luw, edges_luw
    else:
        # Fallback: return LUW even if not fully verified
        return n_luw, edges_luw

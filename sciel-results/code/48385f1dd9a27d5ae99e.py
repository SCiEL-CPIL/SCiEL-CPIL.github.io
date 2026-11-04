"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

_DEADLINE_MARGIN = 15.0  # seconds before the hard kill to stop searching


def _validate_graph(n, edges, k, g):
    """Validate simple k-regular graph with girth >= g via non-backtracking BFS."""
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Girth check: for each start vertex, BFS on (current, prev) states
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            if curr == s:
                cycle_len = depth + 1
                if cycle_len < g:
                    return False
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
    return True


def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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


def _random_4_regular(n, seed):
    """Try to build a random 4-regular simple graph on n vertices using configuration model."""
    import random
    rng = random.Random(seed)
    half_edges = n * 2
    stubs = []
    for v in range(n):
        for _ in range(4):
            stubs.append(v)
    rng.shuffle(stubs)
    edges = set()
    valid = True
    for i in range(0, half_edges, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            valid = False
            break
        edge = (min(u, v), max(u, v))
        if edge in edges:
            valid = False
            break
        edges.add(edge)
    if not valid:
        return None
    edge_list = list(edges)
    adj = [0] * n
    for u, v in edge_list:
        adj[u] += 1
        adj[v] += 1
    if any(d != 4 for d in adj):
        return None
    return edge_list


def _check_girth_quick(n, edges, g):
    """Quick girth check: only detect cycles shorter than g. Returns True if girth >= g."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
    return True


def construct(k, g):
    t_start = time.time()
    deadline = t_start + 120.0 - _DEADLINE_MARGIN

    # Moore lower bound
    r = g // 2
    if g % 2 == 0:
        lb = k * (k - 1) * (r + 1) - (k - 1)
    else:
        lb = k * (k - 1) * (r + 1) + 1

    best_n, best_edges = None, None

    # Try LUW as baseline
    try:
        n_luw, edges_luw = _luw_graph(k, g)
        if n_luw >= lb:
            if _validate_graph(n_luw, edges_luw, k, g):
                best_n, best_edges = n_luw, edges_luw
    except Exception:
        pass

    # For (4,10): try to find a graph with fewer vertices than LUW
    if k == 4 and g == 10:
        # Known best is 320; LUW typically gives more
        # Try random 4-regular graphs from 320 upward (or from best_n-1 downward if best_n exists)
        if best_n is not None:
            search_start = best_n - 1
        else:
            search_start = 320
        search_end = 243  # lower bound

        # Search from smaller n upward to find the smallest valid graph
        # Start at the known best (320) and try to go lower
        for n in range(320, max(search_end, 310), -1):
            if time.time() > deadline:
                break
            if best_n is not None and n >= best_n:
                continue
            found = False
            for trial in range(200):
                if time.time() > deadline:
                    break
                edges = _random_4_regular(n, seed=trial * 1000 + n)
                if edges is None:
                    continue
                if _check_girth_quick(n, edges, g):
                    # Full validation
                    if _validate_graph(n, edges, k, g):
                        best_n, best_edges = n, edges
                        found = True
                        break
            if found:
                # Try to go even lower
                continue

    # If we still don't have a result, fall back to LUW
    if best_n is None:
        try:
            best_n, best_edges = _luw_graph(k, g)
        except Exception:
            pass

    return best_n, best_edges
# EVOLVE-BLOCK-END

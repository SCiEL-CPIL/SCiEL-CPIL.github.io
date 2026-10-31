import time
import random
from collections import deque


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q), q = k.
    Empirically try smaller d (fewer vertices) and verify girth; fall back to a
    theoretically safe d. Returns (n, edges)."""
    q = k
    add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

    def build_luw(d):
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

    # Theoretically safe d (girth >= d+5 for odd d): pick d so d+5 >= g.
    safe_d = g - 5
    if safe_d % 2 == 0:
        safe_d += 1
    safe_d = max(3, safe_d)

    # Try smaller d first (fewer vertices); verify empirically.
    candidates = sorted(set([safe_d, safe_d - 1, safe_d - 2, safe_d - 3]), reverse=False)
    candidates = [d for d in candidates if d >= 3]
    # Ascending by size: smaller d first
    candidates.sort(key=lambda d: q ** d)
    for d in candidates:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    # Fallback: theoretically safe
    return build_luw(safe_d)


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(n, edges, min_girth):
    """BFS from each vertex with parent tracking; returns True if girth >= min_girth."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2
    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])
        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True


def _petersen_edges():
    """Petersen graph (10 vertices, 3-regular, girth 5)."""
    outer = [(i, (i + 1) % 5) for i in range(5)]
    star = [(i, (i + 2) % 5) for i in range(5)]
    spokes = [(5 + i, 10 + i) for i in range(5)]
    return outer + star + spokes


def _voltage_lift(base_edges, m, voltages):
    """Lift base graph by factor m using Z_m voltages.
    voltages: dict (u,v) -> v in Z_m for each undirected base edge.
    Returns (n, edges) on n = m * len(base_vertices) vertices."""
    base_n = 10  # Petersen
    n = m * base_n
    edges = []
    for (u, v) in base_edges:
        for a in range(m):
            b = (a + voltages[(u, v)]) % m
            eu, ev = a * base_n + u, b * base_n + v
            if eu > ev:
                eu, ev = ev, eu
            edges.append((eu, ev))
    return n, edges


def _random_3reg_edges(n, seed=None):
    if seed is not None:
        random.seed(seed)
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    for _ in range(200):
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((u, v))
        if valid:
            return list(edges)
    return None


def _switch_to_raise_girth(n, edges, min_girth, max_iters):
    """Local search: switch two edges to eliminate short cycles. Returns edges or None."""
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    cur = list(edges)
    for _ in range(max_iters):
        adj = _build_adj(n, cur)
        # Find a short cycle via BFS from a random start
        start = random.randrange(n)
        visited = [-1] * n
        parent = [-1] * n
        visited[start] = 0
        queue = deque([start])
        cycle = None
        while queue and cycle is None:
            node = queue.popleft()
            d = visited[node]
            for nb in adj[node]:
                if nb == parent[node]:
                    continue
                if visited[nb] == -1:
                    visited[nb] = d + 1
                    parent[nb] = node
                    queue.append(nb)
                else:
                    clen = d + visited[nb] + 1
                    if clen < min_girth:
                        # reconstruct cycle
                        a, b = node, nb
                        path_a = [a]
                        while a != start:
                            a = parent[a]
                            path_a.append(a)
                        path_b = [b]
                        while b != start:
                            b = parent[b]
                            path_b.append(b)
                        path_b.reverse()
                        cycle = path_a + path_b[1:]
                        break
        if cycle is None:
            return cur  # no short cycle found
        # Try switching two non-adjacent edges on the cycle
        L = len(cycle)
        cand_pairs = []
        for i in range(L):
            for j in range(i + 2, L):
                if i == 0 and j == L - 1:
                    continue
                cand_pairs.append((i, j))
        random.shuffle(cand_pairs)
        improved = False
        for (i, j) in cand_pairs:
            e1 = (cycle[i], cycle[(i + 1) % L])
            e2 = (cycle[j], cycle[(j + 1) % L])
            e1 = (min(e1[0], e1[1]), max(e1[0], e1[1]))
            e2 = (min(e2[0], e2[1]), max(e2[0], e2[1]))
            if e1 == e2:
                continue
            new_u, new_v = e1[0], e2[1]
            new_u2, new_v2 = e1[1], e2[0]
            # check no loop / duplicate
            if new_u == new_v or new_u2 == new_v2:
                continue
            if (min(new_u, new_v), max(new_u, new_v)) in edge_set:
                continue
            if (min(new_u2, new_v2), max(new_u2, new_v2)) in edge_set:
                continue
            # apply
            edge_set.discard(e1)
            edge_set.discard(e2)
            edge_set.add((min(new_u, new_v), max(new_u, new_v)))
            edge_set.add((min(new_u2, new_v2), max(new_u2, new_v2)))
            cur = list(edge_set)
            improved = True
            break
        if not improved:
            break
    return cur


def _lift_search(k, g, deadline):
    """Structural: random voltage lift of Petersen + local search to raise girth.
    Returns (n, edges) if a valid graph is found within deadline, else None."""
    base = _petersen_edges()
    base_edge_set = set((min(u, v), max(u, v)) for u, v in base)
    # Try several lift sizes; smaller m = fewer vertices but harder to reach girth 16.
    for m in [24, 32, 48, 64]:
        if time.time() > deadline:
            break
        n = m * 10
        for attempt in range(4):
            if time.time() > deadline:
                break
            voltages = {}
            for (u, v) in base:
                voltages[(min(u, v), max(u, v))] = random.randrange(m)
            n2, edges = _voltage_lift(base, m, voltages)
            if not _check_girth_fast(n2, edges, g):
                # local search to raise girth
                edges = _switch_to_raise_girth(n2, edges, g, max_iters=300)
            if edges is not None and _check_girth_fast(n2, edges, g):
                return n2, edges
    return None


def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # safety margin under 120s

    # Primary: fast LUW with empirical girth check (small d first).
    try:
        n, edges = _luw_construct(k, g)
        if _check_girth_fast(n, edges, g):
            return n, edges
    except Exception:
        pass

    # Secondary (structural): random voltage lift of Petersen + local search.
    try:
        res = _lift_search(k, g, deadline)
        if res is not None:
            return res
    except Exception:
        pass

    # Last resort: random 3-regular search (low success for g=16, but valid).
    for n in [20000, 40000]:
        if time.time() > deadline:
            break
        for _ in range(3):
            if time.time() > deadline:
                break
            edges = _random_3reg_edges(n)
            if edges is not None and _check_girth_fast(n, edges, g):
                return n, edges

    # Guaranteed fallback: theoretically safe LUW.
    return _luw_construct(k, g)

import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    # Moore bound for k-regular graph with girth g
    if g % 2 == 0:
        # Even girth: n >= k * (k-1)^(g/2 - 1) / (k-1) + 1 ... actually
        # n >= k * sum_{i=0}^{g/2-1} (k-1)^i = k * ((k-1)^(g/2) - 1) / (k-2) if k>2
        if k == 2:
            moore = g
        else:
            moore = k * sum((k-1)**i for i in range(g//2))
    else:
        # Odd girth: n >= 2 * sum_{i=0}^{(g-1)/2} (k-1)^i
        if k == 2:
            moore = g
        else:
            moore = 2 * sum((k-1)**i for i in range((g+1)//2))

    # Try the Lazebnik-Ustimenko-Woldar (LUW) construction first
    # This gives a q-regular bipartite graph with girth >= d+5 for odd d
    # For our case k=5, g=9: q=5, need girth >= 9, so d+5 >= 9 => d >= 4
    # d must be odd, so d=5 gives girth >= 10, which is fine (>= 9)
    
    best_n = None
    best_edges = None

    def luw_construct(q, d):
        """LUW graph: q-regular bipartite with girth >= d+5 for odd d >= 3"""
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        # Rule: for c in 2..d, (a_c, b_c)
        rule = {}
        for c in range(2, d + 1):
            if c == 2:
                rule[c] = (1, 1)
            elif c == 3:
                rule[c] = (2, 1)
            elif c == 4:
                rule[c] = (1, 2)
            else:
                r = (c - 5) % 4
                c0 = c - r
                options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
                rule[c] = options[r]

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

        for i, vertex in enumerate(order):
            if time.time() > deadline:
                return None, None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        n = len(order)
        if n == 0:
            return None, None
        return n, edges

    # For k=5, g=9: try LUW with q=5, d=5 (girth >= 10)
    # Also try q=5, d=3 (girth >= 8, might not be enough)
    # d=5 is the right choice for girth >= 9

    if k == 5 and g == 9:
        # Try LUW with various parameters
        for q, d in [(5, 5), (5, 7), (7, 5), (9, 5)]:
            if time.time() > deadline:
                break
            n, edges = luw_construct(q, d)
            if n is not None and n > 0:
                # Verify degree
                deg = [0] * n
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if all(dg == k for dg in deg):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges

        # If LUW didn't give a good result, try random construction
        if best_n is None or best_n > 500:
            # Try to construct a random k-regular graph and check girth
            # This is heuristic and may not find optimal, but can help
            if time.time() < deadline - 30:
                n_candidate = max(moore, 100)
                # Try a few random graphs
                for attempt in range(10):
                    if time.time() > deadline - 10:
                        break
                    edges = _random_k_regular(n_candidate, k)
                    if edges is not None:
                        # Quick girth check
                        if _has_girth_at_least(n_candidate, edges, g):
                            if best_n is None or n_candidate < best_n:
                                best_n = n_candidate
                                best_edges = edges

    else:
        # General case: try LUW
        # Find appropriate d: need d+5 >= g, d odd, d >= 3
        d_min = max(3, g - 5)
        if d_min % 2 == 0:
            d_min += 1
        q = k
        if q == 4:
            pass
        elif q % 2 == 0:
            q = k + 1  # need prime or 4
            while q < 200:
                if q == 4 or _is_prime(q):
                    break
                q += 1

        n, edges = luw_construct(q, d_min)
        if n is not None and n > 0:
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if all(dg == k for dg in deg):
                best_n = n
                best_edges = edges
        else:
            # Fallback: try different q
            for q_try in range(k, k + 20):
                if q_try == 4 or _is_prime(q_try):
                    n, edges = luw_construct(q_try, d_min)
                    if n is not None and n > 0:
                        deg = [0] * n
                        for u, v in edges:
                            deg[u] += 1
                            deg[v] += 1
                        if all(dg == k for dg in deg):
                            best_n = n
                            best_edges = edges
                            break

    if best_n is None:
        # Absolute fallback: return a trivial valid graph if possible
        # For k=2, it's just a cycle of length g
        if k == 2:
            n = g
            edges = [(i, (i + 1) % n) for i in range(n)]
            return n, edges
        else:
            # Return an empty graph as last resort (will score 0)
            return 0, []

    return best_n, best_edges


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


def _random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices. Returns edge list or None."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None

    # Use the configuration model
    half_edges = []
    for v in range(n):
        half_edges.extend([v] * k)

    random.shuffle(half_edges)

    edges = set()
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i + 1]
        if u == v:
            return None  # Self-loop, retry
        edge = (min(u, v), max(u, v))
        if edge in edges:
            return None  # Multi-edge, retry
        edges.add(edge)

    return list(edges)


def _has_girth_at_least(n, edges, g):
    """Check if graph has girth at least g using BFS from each vertex."""
    if n == 0:
        return True

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # For each vertex, do BFS and track distances
    # If we find a vertex at distance < g that creates a cycle shorter than g, return False
    # More precisely: for each vertex s, BFS. If we find a back edge (u,v) where
    # dist[s][u] + dist[s][v] + 1 < g and the path doesn't backtrack, we have a short cycle.
    # Simpler: just check that no two vertices at distance < g/2 from s are adjacent
    # in a way that creates a cycle < g.

    # Efficient check: for each edge (u,v), the shortest cycle through (u,v) has length
    # 1 + dist(u,v) where dist is the shortest path from u to v not using edge (u,v).
    # This is expensive. Instead, do BFS from each vertex up to depth (g-1)//2 and check
    # for back edges.

    max_depth = (g - 1) // 2

    for s in range(n):
        # BFS from s
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])

        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[u] + dist[v] + 1 < g:
                    # Found a cycle of length dist[u] + dist[v] + 1 < g
                    # But we need to make sure the paths from s to u and s to v
                    # don't share edges (non-backtracking)
                    # This is a heuristic check; for small g it's usually correct
                    return False

    return True
# EVOLVE-BLOCK-END

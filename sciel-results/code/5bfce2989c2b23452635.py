import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5 seconds margin for safety

    # Moore bound
    if g % 2 == 0:
        r = g // 2 - 1
        n_moore = 2 * sum([1] + [k * (k-1)**i for i in range(r)])
    else:
        r = (g - 1) // 2
        n_moore = 1 + k * sum([(k-1)**i for i in range(r)])

    # Fallback: LUW construction from seed
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # Try greedy construction with multiple n values
    for n_attempt in range(n_moore, min(n_moore + 100, 500), 1):
        if time.time() > deadline - 10:
            break
        success, edges = try_greedy(k, g, n_attempt, deadline - 5)
        if success:
            return n_attempt, edges

    # Fallback to LUW
    return luw_construct(k, g)

def try_greedy(k, g, n, deadline):
    """Randomized greedy with efficient BFS-based distance checks."""
    adj = [set() for _ in range(n)]
    remaining_deg = [k] * n
    total_edges = k * n // 2
    edges_added = 0

    def bfs_distances(source, adj, limit):
        """Return dict of vertex -> distance (capped at limit) from source."""
        dist = {source: 0}
        queue = [source]
        qi = 0
        while qi < len(queue):
            node = queue[qi]
            qi += 1
            d = dist[node]
            if d >= limit - 1:
                continue
            for nei in adj[node]:
                if nei not in dist:
                    dist[nei] = d + 1
                    queue.append(nei)
        return dist

    max_attempts = 20000
    attempts = 0

    while edges_added < total_edges:
        if time.time() > deadline:
            return False, []
        attempts += 1
        if attempts > max_attempts:
            return False, []

        # Pick vertex with max remaining degree
        max_rem = max(remaining_deg)
        if max_rem == 0:
            break

        cands_u = [i for i in range(n) if remaining_deg[i] == max_rem]
        u = random.choice(cands_u)

        # BFS from u to get all distances up to g-2
        dist_u = bfs_distances(u, adj, g - 1)

        # Find valid v: not u, not adjacent, remaining_deg > 0, dist >= g-1 or unreachable
        valid_v = []
        for v in range(n):
            if v == u or v in adj[u] or remaining_deg[v] == 0:
                continue
            d = dist_u.get(v, g)  # if not reached, treat as far
            if d >= g - 1:
                valid_v.append(v)

        if not valid_v:
            return False, []

        v = random.choice(valid_v)
        adj[u].add(v)
        adj[v].add(u)
        remaining_deg[u] -= 1
        remaining_deg[v] -= 1
        edges_added += 1

    # Verify regularity
    for i in range(n):
        if len(adj[i]) != k:
            return False, []

    # Girth check: for each vertex, BFS and ensure no edge back to start at dist < g-1
    for u in range(n):
        dist = {u: 0}
        queue = [u]
        qi = 0
        while qi < len(queue):
            node = queue[qi]
            qi += 1
            d = dist[node]
            if d >= g - 1:
                continue
            for nei in adj[node]:
                if nei == u:
                    if d + 1 < g:
                        return False, []
                    continue
                if nei not in dist:
                    dist[nei] = d + 1
                    queue.append(nei)

    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    return True, edges

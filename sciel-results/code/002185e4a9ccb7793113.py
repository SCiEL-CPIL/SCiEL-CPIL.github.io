import time
import random
import sys

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph (large but valid)
    def luw_construct(k, g):
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

    # Moore bound calculation
    D = (g - 2) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(D))

    deadline = time.time() + 110
    best_n = float('inf')
    best_edges = None

    # Helper to check girth using non-backtracking walks
    def has_girth_at_least(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (v, u) in edge_set:
                return False
            edge_set.add((u, v))
        # Check degree
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # BFS from each vertex with non-backtracking
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    # Helper to generate random k-regular graph
    def random_regular_graph(n, k):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = []
        valid = True
        seen = set()
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            if (u, v) in seen or (v, u) in seen:
                valid = False
                break
            seen.add((u, v))
            edges.append((u, v))
        if valid:
            return edges
        return None

    # Search for smaller graphs
    # Try n=320 first (known cage size), then smaller
    search_ns = [320]
    for n in range(319, moore_bound - 1, -1):
        search_ns.append(n)
    search_ns = list(dict.fromkeys(search_ns))  # Remove duplicates, keep order

    for n in search_ns:
        if time.time() > deadline:
            break
        # More attempts for n=320
        attempts = 3000 if n == 320 else 300
        for _ in range(attempts):
            if time.time() > deadline:
                break
            edges = random_regular_graph(n, k)
            if edges is None:
                continue
            if has_girth_at_least(n, edges, g):
                return n, edges

    # Fallback to LUW
    return luw_construct(k, g)

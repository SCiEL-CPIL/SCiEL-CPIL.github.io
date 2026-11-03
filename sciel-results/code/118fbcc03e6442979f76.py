"""Optimized construction for 3-regular graph of girth 16 (cage problem)."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = 125  # safety margin before 150s limit

    def luw_construction(k, g):
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

    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        half = min_girth // 2
        for start in range(n):
            dist = [-1] * n
            par = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= half:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        par[v] = u
                        q.append(v)
                    elif v != par[u] and dist[u] + 1 < dist[v]:
                        a, b = u, v
                        while a != b:
                            if dist[a] > dist[b]:
                                a = par[a]
                            elif dist[b] > dist[a]:
                                b = par[b]
                            else:
                                a = par[a]
                                b = par[b]
                        cycle_len = dist[u] + dist[v] - 2 * dist[a] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def random_3_regular(n):
        if n % 2 != 0 or n < 4:
            return None
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        edges = []
        adj = [set() for _ in range(n)]
        for i in range(0, n * 3, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in adj[u]:
                return None
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        return edges

    luw_n, luw_edges = luw_construction(k, g)
    best_n, best_edges = luw_n, luw_edges

    if g % 2 == 0:
        h = g // 2 - 1
        moore_lb = 2 * sum(2**i for i in range(h + 1))
    else:
        h = (g - 1) // 2 - 1
        moore_lb = 1 + 3 * sum(2**i for i in range(h + 1))

    search_from = max(moore_lb, 500)
    for n in range(search_from, min(best_n, search_from + 1500), 5):
        if time.time() - start_time > deadline:
            break
        if n % 2 != 0:
            n += 1
        for attempt in range(20):
            if time.time() - start_time > deadline:
                break
            edges = random_3_regular(n)
            if edges is None:
                continue
            if check_girth(n, edges, g):
                if best_n is None or n < best_n:
                    best_n, best_edges = n, edges
                break

    return best_n, best_edges
# EVOLVE-BLOCK-END

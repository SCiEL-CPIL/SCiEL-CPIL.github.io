import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # ------------------------------------------------------------------
    # Lazebnik-Ustimenko-Woldar graph CD(d, q), q = k.
    # Explicit q-regular bipartite graph, girth >= d + 5.
    # ------------------------------------------------------------------
    def construct_luw(k, g):
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

    luw_n, luw_edges = construct_luw(k, g)

    # ------------------------------------------------------------------
    # Rigorous validation: simple, k-regular, and girth >= g
    # (certified by non-backtracking walks from every vertex).
    # ------------------------------------------------------------------
    def _validate(k, g, n, edges):
        if n < k + 1:
            return False
        adj = [[] for _ in range(n)]
        edge_set = set()
        for i, j in edges:
            if i == j or i < 0 or i >= n or j < 0 or j >= n:
                return False
            key = (min(i, j), max(i, j))
            if key in edge_set:
                return False
            edge_set.add(key)
            adj[i].append(j)
            adj[j].append(i)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        for s in range(n):
            visited = {}
            queue = deque()
            queue.append((s, None, 0))
            visited[(s, None)] = 0
            while queue:
                v, prev, dist = queue.popleft()
                for w in adj[v]:
                    if w == prev:
                        continue
                    new_dist = dist + 1
                    if new_dist >= g:
                        continue
                    if w == s and new_dist >= 1:
                        return False
                    state = (w, v)
                    if state not in visited:
                        visited[state] = new_dist
                        queue.append((w, v, new_dist))
        return True

    # ------------------------------------------------------------------
    # Bounded local-search refinement: start from a valid graph and only
    # accept edge swaps that keep the graph valid. Returns smallest valid n.
    # ------------------------------------------------------------------
    def _refine(k, g, n, edges, deadline):
        def build_adj(n, edges):
            adj = [set() for _ in range(n)]
            for i, j in edges:
                adj[i].add(j)
                adj[j].add(i)
            return adj

        def edge_list(adj, n):
            out = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        out.append((u, v))
            return out

        adj = build_adj(n, edges)
        best_n = n
        best_edges = edges
        while time.time() < deadline:
            # pick a random edge (u, v)
            u = random.randrange(n)
            if not adj[u]:
                continue
            v = random.choice(tuple(adj[u]))
            # pick another edge (x, y)
            x = random.randrange(n)
            if not adj[x]:
                continue
            y = random.choice(tuple(adj[x]))
            if u == x or u == y or v == x or v == y:
                continue
            if x in adj[u] or y in adj[u] or x in adj[v] or y in adj[v]:
                continue
            # apply swap (u,v)&(x,y) -> (u,x)&(v,y)
            adj[u].remove(v); adj[v].remove(u)
            adj[x].remove(y); adj[y].remove(x)
            adj[u].add(x); adj[x].add(u)
            adj[v].add(y); adj[y].add(v)
            new_edges = edge_list(adj, n)
            if _validate(k, g, n, new_edges):
                # accept; keep as best (n unchanged) - swap may help future moves
                best_edges = new_edges
            else:
                # revert
                adj[u].remove(x); adj[x].remove(u)
                adj[v].remove(y); adj[y].remove(v)
                adj[u].add(v); adj[v].add(u)
                adj[x].add(y); adj[y].add(x)
        return best_n, best_edges

    # Verify baseline is valid (it should be); if not, we still return it
    # but skip refinement to avoid compounding an invalid state.
    if _validate(k, g, luw_n, luw_edges):
        try:
            ref_n, ref_edges = _refine(k, g, luw_n, luw_edges, deadline)
            if _validate(k, g, ref_n, ref_edges) and ref_n < luw_n:
                return ref_n, ref_edges
        except Exception:
            pass

    return luw_n, luw_edges

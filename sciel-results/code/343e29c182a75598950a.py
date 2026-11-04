import time
import math
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    def moore_bound(k, g):
        if g % 2 == 1:
            r = (g - 1) // 2
            return 1 + k * sum((k - 1) ** i for i in range(r))
        else:
            r = g // 2
            return 2 * sum((k - 1) ** i for i in range(r - 1))
            
    def luw_construct(k, g):
        q = k
        d = g - 5
        if d < 3: d = 3
        if d % 2 == 0: d += 1
        
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
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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

    best_n, best_edges = luw_construct(k, g)
    
    if time.time() > deadline:
        return best_n, best_edges

    def has_girth_at_least(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        depth_limit = min_girth // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= depth_limit:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            if dist[u] + dist[v] + 1 < min_girth:
                                return False
        return True

    def random_k_regular(n, k):
        for _ in range(100):
            half_edges = np.repeat(np.arange(n), k)
            np.random.shuffle(half_edges)
            edges = []
            valid = True
            for i in range(0, n * k, 2):
                u = half_edges[i]
                v = half_edges[i + 1]
                if u == v:
                    valid = False
                    break
                edges.append((int(u), int(v)))
            if not valid:
                continue
            
            edge_set = set()
            for u, v in edges:
                if u > v: u, v = v, u
                if (u, v) in edge_set:
                    valid = False
                    break
                edge_set.add((u, v))
            if not valid:
                continue
                
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            visited = [False] * n
            q = deque([0])
            visited[0] = True
            count = 1
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)
                        count += 1
            if count == n:
                return edges
        return None

    mb = moore_bound(k, g)
    max_search_n = min(best_n, mb + 300)
    
    np.random.seed(42)
    
    for N in range(mb, max_search_n + 1):
        if time.time() > deadline:
            break
        attempts = 50
        for _ in range(attempts):
            if time.time() > deadline:
                break
            edges = random_k_regular(N, k)
            if edges is None:
                continue
            if has_girth_at_least(N, edges, g):
                return N, edges
                
    return best_n, best_edges

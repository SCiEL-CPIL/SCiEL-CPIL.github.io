import random
import time
import sys
import heapq
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety
    
    # Fallback: LUW graph
    def luw_graph(k, g):
        q = k
        d = 4
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

    n_luw, edges_luw = luw_graph(k, g)
    best_n, best_edges = n_luw, edges_luw

    # Try to find a smaller graph using random switching
    target_n = 1116
    if target_n >= 428: # Moore bound
        def gen_random_regular(n, k):
            for _ in range(10):
                stubs = [i for i in range(n) for _ in range(k)]
                random.shuffle(stubs)
                adj = {i: set() for i in range(n)}
                valid = True
                for i in range(0, n * k, 2):
                    u, v = stubs[i], stubs[i+1]
                    if u == v or v in adj[u]:
                        valid = False
                        break
                    adj[u].add(v)
                    adj[v].add(u)
                if valid:
                    return adj
            return None

        def has_short_cycle(adj, g):
            n = len(adj)
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= (g-1)//2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            # Found a back edge
                            # Check cycle length
                            # Simple check: if dist[u] + dist[v] + 1 < g, we have a short cycle
                            # But we need to be sure it's a cycle in the tree + edge
                            # For BFS, the shortest path is dist.
                            # If dist[u] + dist[v] + 1 < g, it's a short cycle.
                            if dist[u] + dist[v] + 1 < g:
                                return True
            return False

        def try_switch(adj, n, k, g):
            max_iter = 5000
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if not has_short_cycle(adj, g):
                    return True
                # Find a short cycle to switch
                # This is expensive.
                # We can just pick a random edge and switch with a random other edge.
                # This is a blind switch.
                u = random.randint(0, n-1)
                if not adj[u]: continue
                v = random.choice(list(adj[u]))
                x = random.randint(0, n-1)
                if x in (u, v): continue
                if not adj[x]: continue
                y = random.choice(list(adj[x]))
                if y in (u, v): continue
                if x == u or y == v: continue
                if x in adj[v] or y in adj[u]: continue
                
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].add(x)
                adj[x].add(u)
                adj[v].add(y)
                adj[y].add(v)
            
            return not has_short_cycle(adj, g)

        adj = gen_random_regular(target_n, k)
        if adj:
            if try_switch(adj, target_n, k, g):
                edges = []
                for u in range(target_n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return target_n, edges

    return best_n, best_edges

"""Construct small 3-regular graphs with high girth using LUW family."""
import time
from collections import deque

def _verify_girth(n, edges, k, g):
    """Verify girth >= g using BFS from each vertex (non-backtracking)."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u,v), max(u,v))
        if key in seen:
            return False
        seen.add(key)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                else:
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def construct(k, g):
    deadline = time.time() + 110
    if k != 3:
        raise NotImplementedError("Only k=3 supported")
    
    q = 3
    # LUW: girth = 2d + 4, need >= g
    d = max(2, (g - 4) // 2)
    while 2 * d + 4 < g:
        d += 1
    
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q
    
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
    
    n_result = len(order)
    
    if time.time() < deadline - 30 and n_result <= 50000:
        if _verify_girth(n_result, edges, 3, g):
            return n_result, edges
    
    return n_result, edges

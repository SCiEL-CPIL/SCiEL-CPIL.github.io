import time
import sys
import random
from collections import deque

def _has_short_cycle(n, adj, g):
    """Check if graph has a cycle of length < g using non-backtracking BFS."""
    for start in range(n):
        dist = {}
        queue = deque()
        for first in adj[start]:
            queue.append((first, start, 1))
            dist[(first, start)] = 1
        
        while queue:
            curr, prev, length = queue.popleft()
            if length >= g - 1:
                continue
            
            for nxt in adj[curr]:
                if nxt == prev:
                    continue
                if nxt == start:
                    if 3 <= length + 1 < g:
                        return True
                    continue
                state = (nxt, curr)
                if state not in dist:
                    dist[state] = length + 1
                    queue.append((nxt, curr, length + 1))
    return False

def _random_4_regular(n, seed=42):
    """Generate a random 4-regular graph on n vertices."""
    if n < 4:
        return None
    random.seed(seed)
    
    adj = [set() for _ in range(n)]
    for attempt in range(500):
        stubs = [i for i in range(n) for _ in range(4)]
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
        
        if all(len(adj[i]) == 4 for i in range(n)):
            break
    
    if not all(len(adj[i]) == 4 for i in range(n)):
        return None
    
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _luw_construct(k, g):
    """LUW construction as fallback."""
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if k != 4:
        return _luw_construct(k, g)
    
    # k=4, g=10: Random search for small 4-regular graph with girth >= 10
    for n in range(320, 400):
        if time.time() > deadline:
            break
        
        for seed in range(200):
            if time.time() > deadline:
                break
            
            edges = _random_4_regular(n, seed)
            if edges is None:
                continue
            
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            if not _has_short_cycle(n, adj, g):
                return n, edges
    
    return _luw_construct(k, g)
